# Muse Stress Monitor

Real-time EEG stress monitoring application using Muse headband data.

## Prerequisites

- Python 3.8+
- Muse headband
- muselsl package configured

## Installation

```bash
pip install dearpygui numpy scipy pylsl muselsl
```

## Configuration

Update `MUSE_ADDRESS` in code with your Muse headband's address:
```python
MUSE_ADDRESS = "YOUR-MUSE-ADDRESS"
```

To find your Muse address:
```bash
muselsl list
```

## Usage

1. Power on Muse headband
2. Run application:
```bash
python muse_monitor.py
```

## GUI Operation

1. **Main Window Layout**:
   - Left: EEG wave visualizations
   - Right: Controls and results

2. **Recording Session**:
   - Click "Start Recording"
   - Ensure good headband contact
   - Stay still for 30-second recording
   - View real-time stress levels

3. **Interpreting Results**:
   - Stress level: 0-100%
   - Color indicators:
     - Green: Very relaxed
     - Blue: Relaxed
     - Yellow: Mild stress
     - Red: High stress
   - Alpha/Beta ratio > 1.0: Relaxation
   - Alpha/Beta ratio < 1.0: Stress/Activity

## Troubleshooting

1. No EEG stream:
   - Verify Muse is powered on
   - Check Bluetooth connection
   - Confirm correct Muse address

2. Poor signal:
   - Clean sensors
   - Adjust headband position
   - Check battery level
