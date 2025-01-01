import dearpygui.dearpygui as dpg
import asyncio
import numpy as np
from scipy import signal
from pylsl import StreamInlet, resolve_stream
import time
import threading
import collections
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
RECORD_DURATION = 30
MUSE_ADDRESS = "170A1E6D-C386-2E20-6012-76E4C5586FD7"
SAMPLING_RATE = 256
DISPLAY_TIME = 5
BUFFER_SIZE = SAMPLING_RATE * DISPLAY_TIME
DISPLAY_SCALE = 20
WINDOW_SIZE = SAMPLING_RATE * 2
AVERAGE_WINDOW = 50
UPDATE_INTERVAL = 0.033  # ~30 FPS


class MuseMonitor:
    def __init__(self):
        self.data_lock = threading.Lock()
        self.raw_buffer = collections.deque(maxlen=BUFFER_SIZE)
        self.alpha_buffer = collections.deque(maxlen=BUFFER_SIZE)
        self.beta_buffer = collections.deque(maxlen=BUFFER_SIZE)
        self.theta_buffer = collections.deque(maxlen=BUFFER_SIZE)
        self.gamma_buffer = collections.deque(maxlen=BUFFER_SIZE)
        self.stress_scores = collections.deque(maxlen=AVERAGE_WINDOW)
        self.is_recording = False
        self.recording_complete = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.running = True

    def update_status(self, message):
        dpg.set_value("status_message", message)
        logger.info(message)

    def calculate_band_power(self, data, low_freq, high_freq):
        if len(data) < WINDOW_SIZE:
            return 0

        with self.data_lock:
            data = np.array(list(data)[-WINDOW_SIZE:])

        freqs, psd = signal.welch(data, fs=SAMPLING_RATE, nperseg=min(256, len(data)))
        low_idx = np.argmax(freqs >= low_freq)
        high_idx = np.argmax(freqs >= high_freq)
        return np.mean(psd[low_idx:high_idx])

    def calculate_stress_from_ratio(self, ratio):
        if ratio <= 0:
            return 100
        return 100 - (100 * (ratio / (ratio + 1)))

    def calculate_current_stress(self):
        if len(self.beta_buffer) < WINDOW_SIZE:
            return 50.0

        alpha_power = self.calculate_band_power(self.alpha_buffer, 8, 13)
        beta_power = self.calculate_band_power(self.beta_buffer, 13, 30)

        if beta_power > 0:
            ratio = alpha_power / beta_power
            stress = self.calculate_stress_from_ratio(ratio)
            with self.data_lock:
                self.stress_scores.append(stress)
                return np.mean(list(self.stress_scores))
        return 50.0

    def get_stress_state(self, stress_level):
        if stress_level <= 25:
            return "Very Relaxed"
        elif stress_level <= 65:
            return "Relaxed"
        elif stress_level <= 80:
            return "Mildly Stressed/Activated"
        return "Highly Stressed/Full Activation"

    def get_stress_color(self, stress_level):
        if stress_level <= 25:
            return (0, 255, 0, 255)
        elif stress_level <= 50:
            return (0, 191, 255, 255)
        elif stress_level <= 75:
            return (255, 255, 0, 255)
        return (255, 0, 0, 255)

    async def connect_to_muse(self):
        try:
            self.update_status("Connecting to Muse headband...")
            process = await asyncio.create_subprocess_shell(
                f"muselsl stream --address {MUSE_ADDRESS}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            return process
        except Exception as e:
            self.update_status(f"Failed to connect to Muse: {e}")
            return None

    async def record_eeg(self):
        try:
            streams = resolve_stream('type', 'EEG')
            if not streams:
                self.update_status("No EEG stream found. Check Muse connection.")
                return

            inlet = StreamInlet(streams[0], max_buflen=BUFFER_SIZE)
            start_time = time.time()
            self.is_recording = True
            self.stress_scores.clear()

            while time.time() - start_time < RECORD_DURATION and self.running:
                sample, timestamp = inlet.pull_sample(timeout=1.0)
                if sample is None:
                    continue

                frontal_avg = np.mean([sample[1], sample[2]])
                with self.data_lock:
                    self.raw_buffer.append(frontal_avg)

                if len(self.raw_buffer) >= SAMPLING_RATE:
                    self.process_eeg_data()

                remaining_time = int(RECORD_DURATION - (time.time() - start_time))
                self.update_status(f"Recording... {remaining_time}s remaining")

            self.finalize_recording()

        except Exception as e:
            self.update_status(f"Recording error: {e}")
            logger.error(f"Recording error: {e}", exc_info=True)
        finally:
            self.is_recording = False

    def process_eeg_data(self):
        with self.data_lock:
            raw_data = list(self.raw_buffer)

        theta, alpha, beta, gamma = self.filter_eeg_bands(raw_data)
        if theta is not None:
            with self.data_lock:
                self.theta_buffer.append(theta[-1])
                self.alpha_buffer.append(alpha[-1])
                self.beta_buffer.append(beta[-1])
                self.gamma_buffer.append(gamma[-1])

    def filter_eeg_bands(self, data):
        if len(data) < SAMPLING_RATE:
            return None, None, None, None

        nyquist = SAMPLING_RATE / 2
        data = np.array(data)

        def create_filter(low, high):
            return signal.butter(6, [low / nyquist, high / nyquist], btype='band')

        theta_b, theta_a = create_filter(4, 8)
        alpha_b, alpha_a = create_filter(8, 13)
        beta_b, beta_a = create_filter(13, 30)
        gamma_b, gamma_a = create_filter(30, 100)

        return (
            signal.filtfilt(theta_b, theta_a, data),
            signal.filtfilt(alpha_b, alpha_a, data),
            signal.filtfilt(beta_b, beta_a, data),
            signal.filtfilt(gamma_b, gamma_a, data)
        )

    def finalize_recording(self):
        if len(self.stress_scores) > 0:
            final_stress = np.mean(list(self.stress_scores))
            state = self.get_stress_state(final_stress)

            alpha_power = self.calculate_band_power(self.alpha_buffer, 8, 13)
            beta_power = self.calculate_band_power(self.beta_buffer, 13, 30)
            ratio = alpha_power / beta_power if beta_power > 0 else float('inf')

            result_text = f"""
Final Results:
Stress Level: {final_stress:.1f}%
State: {state}

Analysis:
Alpha/Beta Ratio: {ratio:.2f}
(> 1.0 indicates relaxation, < 1.0 indicates stress)
"""
            dpg.set_value("results_text", result_text)

        self.recording_complete = True
        self.update_status("Recording complete! See results below.")

    def update_plot(self):
        last_update = time.time()

        while self.running and dpg.is_dearpygui_running():
            current_time = time.time()
            if current_time - last_update < UPDATE_INTERVAL:
                time.sleep(UPDATE_INTERVAL / 2)
                continue

            last_update = current_time

            with self.data_lock:
                if len(self.raw_buffer) > 0 and len(self.alpha_buffer) >= 200:
                    self.update_wave_plots()
                    self.update_stress_indicators()

    def update_wave_plots(self):
        x_data = list(range(len(self.raw_buffer)))[-200:]

        for wave_type, buffer in [
            ("theta", self.theta_buffer),
            ("alpha", self.alpha_buffer),
            ("beta", self.beta_buffer),
            ("gamma", self.gamma_buffer)
        ]:
            data = list(buffer)[-200:]
            scaled_data = np.array(data) * DISPLAY_SCALE
            dpg.set_value(f"{wave_type}_series", [x_data, scaled_data])

            if len(scaled_data) > 0:
                y_min, y_max = np.min(scaled_data), np.max(scaled_data)
                margin = (y_max - y_min) * 0.1
                dpg.set_axis_limits(f"y_axis_{wave_type}",
                                    y_min - margin,
                                    y_max + margin)

    def update_stress_indicators(self):
        stress_score = self.calculate_current_stress()
        dpg.set_value("stress_level", stress_score / 100)
        dpg.configure_item("stress_level", overlay=f"{stress_score:.1f}%")

        state = self.get_stress_state(stress_score)
        color = self.get_stress_color(stress_score)
        dpg.configure_item("state_value", default_value=state)
        dpg.configure_item("state_value", color=color)

    def start_recording(self, sender, app_data, user_data):
        self.recording_complete = False

        async def record_session():
            muse_process = await self.connect_to_muse()
            if muse_process:
                await asyncio.sleep(5)  # Allow time for connection
                await self.record_eeg()
                muse_process.terminate()
                await muse_process.wait()

        self.executor.submit(lambda: asyncio.run(record_session()))

    def create_wave_plot(self, label, tag_prefix, color):
        with dpg.plot(label=label, height=150, width=-1):
            with dpg.theme() as wave_theme:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvThemeCol_PlotLines, color)

            dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag=f"x_axis_{tag_prefix}")
            dpg.set_axis_limits(f"x_axis_{tag_prefix}", 1088, 1200)
            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Amplitude",
                                       tag=f"y_axis_{tag_prefix}")
            series = dpg.add_line_series([], [], label=label, parent=y_axis,
                                         tag=f"{tag_prefix}_series")
            dpg.bind_item_theme(series, wave_theme)

    def setup_gui(self):
        dpg.create_context()

        with dpg.window(label="Muse Stress Monitor", tag="main_window"):
            with dpg.group(horizontal=True):
                # Left panel - Wave plots
                with dpg.child_window(width=800, height=700, tag="left_panel"):
                    self.create_wave_plot("Alpha Waves", "alpha", (0, 255, 0, 255))
                    self.create_wave_plot("Beta Waves", "beta", (255, 0, 0, 255))
                    self.create_wave_plot("Theta Waves", "theta", (0, 0, 255, 255))
                    self.create_wave_plot("Gamma Waves", "gamma", (255, 0, 255, 255))

                # Right panel - Controls and Results
                with dpg.child_window(width=350, height=700, tag="right_panel"):
                    with dpg.collapsing_header(label="Controls", default_open=True):
                        dpg.add_text("Status:", tag="status_label")
                        dpg.add_text("Ready", tag="status_message")
                        dpg.add_button(label="Start Recording",
                                       callback=self.start_recording)

                    with dpg.collapsing_header(label="Current State",
                                               default_open=True):
                        dpg.add_text("Stress Level:", tag="stress_level_label")
                        dpg.add_progress_bar(tag="stress_level", default_value=0,
                                             overlay="0%")
                        with dpg.group(horizontal=True):
                            dpg.add_text("State: ", tag="state_label")
                            dpg.add_text("Ready", tag="state_value")

                    with dpg.collapsing_header(label="Session Results",
                                               default_open=True):
                        dpg.add_text("", tag="results_text", wrap=300)

        dpg.create_viewport(title="Muse Stress Monitor", width=1200, height=800)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)

    def run(self):
        self.setup_gui()
        update_thread = threading.Thread(target=self.update_plot, daemon=True)
        update_thread.start()

        try:
            while dpg.is_dearpygui_running():
                dpg.render_dearpygui_frame()
        finally:
            self.running = False
            self.executor.shutdown(wait=True)
            dpg.destroy_context()


if __name__ == "__main__":
    monitor = MuseMonitor()
    monitor.run()