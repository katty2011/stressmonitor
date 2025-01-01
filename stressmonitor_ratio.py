import dearpygui.dearpygui as dpg
import asyncio
import numpy as np
from scipy import signal
from pylsl import StreamInlet, resolve_stream
import time
import threading
import collections

# Constants
RECORD_DURATION = 30
MUSE_ADDRESS = "170A1E6D-C386-2E20-6012-76E4C5586FD7"
SAMPLING_RATE = 256
DISPLAY_TIME = 5
BUFFER_SIZE = SAMPLING_RATE * DISPLAY_TIME
DISPLAY_SCALE = 20
WINDOW_SIZE = SAMPLING_RATE * 2  # 2 seconds of data
AVERAGE_WINDOW = 50  # Number of samples to average

# Global variables
raw_buffer = collections.deque(maxlen=BUFFER_SIZE)
alpha_buffer = collections.deque(maxlen=BUFFER_SIZE)
beta_buffer = collections.deque(maxlen=BUFFER_SIZE)
theta_buffer = collections.deque(maxlen=BUFFER_SIZE)
gamma_buffer = collections.deque(maxlen=BUFFER_SIZE)
stress_scores = collections.deque(maxlen=AVERAGE_WINDOW)
is_recording = False
recording_complete = False


def update_status(message):
    dpg.set_value("status_message", message)
    print(message)


def calculate_band_power(data, low_freq, high_freq):
    """Calculate power in a specific frequency band using Welch's method"""
    if len(data) < WINDOW_SIZE:
        return 0

    data = np.array(list(data)[-WINDOW_SIZE:])
    freqs, psd = signal.welch(data, fs=SAMPLING_RATE, nperseg=min(256, len(data)))

    low_idx = np.argmax(freqs >= low_freq)
    high_idx = np.argmax(freqs >= high_freq)

    band_power = np.mean(psd[low_idx:high_idx])
    return band_power


def calculate_stress_from_ratio(ratio):
    """Convert alpha/beta ratio to stress score consistently"""
    if ratio <= 0:  # Handle invalid ratios
        return 100
    return 100 - (100 * (ratio / (ratio + 1)))


def calculate_current_stress():
    """Calculate current stress using rolling average"""
    if len(beta_buffer) < WINDOW_SIZE:
        return 50.0  # Default value

    alpha_power = calculate_band_power(alpha_buffer, 8, 13)
    beta_power = calculate_band_power(beta_buffer, 13, 30)

    if beta_power > 0:
        ratio = alpha_power / beta_power
        stress = calculate_stress_from_ratio(ratio)
        stress_scores.append(stress)

        # Return average of recent scores
        return np.mean(list(stress_scores))
    return 50.0


def get_stress_state(stress_level):
    """Return state description based on stress level"""
    if stress_level <= 25:
        return "Very Relaxed"
    elif stress_level <= 65:
        return "Relaxed"
    elif stress_level <= 80:
        return "Mildly Stressed/Activated"
    else:
        return "Highly Stressed/Full Activation.\n Your neurons are firing !"


def get_stress_color(stress_level):
    """Return color based on stress level"""
    if stress_level <= 25:
        return (0, 255, 0, 255)  # Green
    elif stress_level <= 50:
        return (0, 191, 255, 255)  # Blue
    elif stress_level <= 75:
        return (255, 255, 0, 255)  # Yellow
    else:
        return (255, 0, 0, 255)  # Red


async def connect_to_muse(address):
    update_status(f"Connecting to Muse headband...")
    process = await asyncio.create_subprocess_shell(
        f"muselsl stream --address {address}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    return process


async def record_eeg():
    global is_recording, recording_complete

    update_status("Looking for EEG stream...")
    streams = resolve_stream('type', 'EEG')

    if not streams:
        update_status("No EEG stream found. Make sure Muse is connected.")
        return

    inlet = StreamInlet(streams[0])
    start_time = time.time()
    is_recording = True
    all_data = []
    stress_scores.clear()  # Clear previous scores

    while time.time() - start_time < RECORD_DURATION:
        sample, timestamp = inlet.pull_sample()
        frontal_avg = np.mean([sample[1], sample[2]])  # Average of AF7 and AF8
        raw_buffer.append(frontal_avg)
        all_data.append(frontal_avg)

        if len(raw_buffer) >= SAMPLING_RATE:
            theta, alpha, beta, gamma = filter_eeg_bands(list(raw_buffer))
            if theta is not None:
                theta_buffer.append(theta[-1])
                alpha_buffer.append(alpha[-1])
                beta_buffer.append(beta[-1])
                gamma_buffer.append(gamma[-1])

        remaining_time = int(RECORD_DURATION - (time.time() - start_time))
        update_status(f"Recording... {remaining_time} seconds remaining")

    is_recording = False

    # Calculate final results using the same stress scores
    if len(stress_scores) > 0:
        final_stress = np.mean(list(stress_scores))
        state = get_stress_state(final_stress)

        # Calculate final powers for display
        alpha_power = calculate_band_power(alpha_buffer, 8, 13)
        beta_power = calculate_band_power(beta_buffer, 13, 30)
        ratio = alpha_power / beta_power if beta_power > 0 else float('inf')

        result_text = f"""
Final Results:
Stress Level: {final_stress:.1f}%
State: {state}

Detailed Analysis:
Alpha/Beta Ratio: {ratio:.2f}
(> 1.0 indicates relaxation, < 1.0 indicates activity/stress)


"""
        # Wave Powers:
       # Alpha(Relaxation): {alpha_power: .2f}
       # Beta(Active / Stress): {beta_power: .2f}
        dpg.set_value("results_text", result_text)

    recording_complete = True
    update_status("Recording complete! See your results below.")


def start_recording(sender, app_data, user_data):
    global recording_complete
    recording_complete = False

    async def record_session():
        muse_process = await connect_to_muse(MUSE_ADDRESS)
        await asyncio.sleep(5)
        await record_eeg()
        muse_process.terminate()
        await muse_process.wait()

    threading.Thread(target=lambda: asyncio.run(record_session())).start()


def filter_eeg_bands(data):
    if len(data) < SAMPLING_RATE:
        return None, None, None, None

    nyquist = SAMPLING_RATE / 2
    data = np.array(data)

    theta_b, theta_a = signal.butter(6, [4 / nyquist, 8 / nyquist], btype='band')
    alpha_b, alpha_a = signal.butter(6, [8 / nyquist, 13 / nyquist], btype='band')
    beta_b, beta_a = signal.butter(6, [13 / nyquist, 30 / nyquist], btype='band')
    gamma_b, gamma_a = signal.butter(6, [30 / nyquist, 100 / nyquist], btype='band')

    theta = signal.filtfilt(theta_b, theta_a, data)
    alpha = signal.filtfilt(alpha_b, alpha_a, data)
    beta = signal.filtfilt(beta_b, beta_a, data)
    gamma = signal.filtfilt(gamma_b, gamma_a, data)

    return theta, alpha, beta, gamma


def create_wave_plot(label, tag_prefix, color):
    with dpg.plot(label=label, height=150, width=-1):
        with dpg.theme() as wave_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvThemeCol_PlotLines, color)

        dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag=f"x_axis_{tag_prefix}")
        dpg.set_axis_limits(f"x_axis_{tag_prefix}", 1088, 1200)
        y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Amplitude", tag=f"y_axis_{tag_prefix}")
        series = dpg.add_line_series([], [], label=label, parent=y_axis, tag=f"{tag_prefix}_series")
        dpg.bind_item_theme(series, wave_theme)


# GUI creation
dpg.create_context()

with dpg.window(label="Muse Stress Monitor", tag="main_window"):
    with dpg.group(horizontal=True):
        # Left column - Wave plots (70% width)
        with dpg.child_window(width=800, height=700, tag="left_panel"):
            create_wave_plot("Alpha Waves (Relaxation)", "alpha", (0, 255, 0, 255))
            create_wave_plot("Beta Waves (Active/Stress)", "beta", (255, 0, 0, 255))
            create_wave_plot("Theta Waves (Deep Relaxation)", "theta", (0, 0, 255, 255))
            create_wave_plot("Gamma Waves (High Cognition)", "gamma", (255, 0, 255, 255))

        # Right column - Controls and Results (30% width)
        with dpg.child_window(width=350, height=700, tag="right_panel"):
            with dpg.collapsing_header(label="Controls", default_open=True):
                dpg.add_text("Status:", tag="status_label")
                dpg.add_text("Ready to start", tag="status_message")
                dpg.add_button(label="Start Recording", callback=start_recording)

            with dpg.collapsing_header(label="Current State", default_open=True):
                dpg.add_text("Stress Level:", tag="stress_level_label")
                dpg.add_progress_bar(tag="stress_level", default_value=0, overlay="0%")
                with dpg.group(horizontal=True):
                    dpg.add_text("State: ", tag="state_label")
                    dpg.add_text("Ready", tag="state_value")

            with dpg.collapsing_header(label="Session Results", default_open=True):
                dpg.add_text("", tag="results_text", wrap=300)


def update_plot():
    while dpg.is_dearpygui_running():
        if len(raw_buffer) > 0:
            x_data = list(range(len(raw_buffer)))[-200:]

            if len(alpha_buffer) >= 200:
                # Update wave plots
                for wave_type, buffer in [
                    ("theta", theta_buffer),
                    ("alpha", alpha_buffer),
                    ("beta", beta_buffer),
                    ("gamma", gamma_buffer)
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

                # Calculate and update current stress level
                stress_score = calculate_current_stress()
                dpg.set_value("stress_level", stress_score / 100)
                dpg.configure_item("stress_level", overlay=f"{stress_score:.1f}%")

                state = get_stress_state(stress_score)
                color = get_stress_color(stress_score)
                dpg.configure_item("state_value", default_value=state)
                dpg.configure_item("state_value", color=color)

        time.sleep(0.1)


# Configure window and start application
dpg.create_viewport(title="Muse Stress Monitor", width=1200, height=800)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("main_window", True)

# Start the plot update thread
threading.Thread(target=update_plot, daemon=True).start()

# Start the GUI event loop
while dpg.is_dearpygui_running():
    dpg.render_dearpygui_frame()

dpg.destroy_context()