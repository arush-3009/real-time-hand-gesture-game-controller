# Hand Gesture Game Control

> Control games using computer vision and hand gestures!

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📖 Overview

A Python application that uses computer vision to detect hand gestures via webcam and translates them into keyboard controls for games like Asphalt 8.

### Key Features

- **Real-time Hand Detection** - MediaPipe-powered hand tracking at ~30 FPS
- **5 Gesture Controls** - Open hand, fist, V-sign, index pointing, steering
- **Adaptive Smoothing** - Anti-flicker algorithm for stable controls
- **Dual Display Modes** - Webcam view or minimal info overlay
- **Configurable** - YAML-based settings for easy customization

## 🎯 Gestures

| Gesture               | Action            | Key |
|-----------------------|-------------------|-----|
| ✋ Open Hand          | Accelerate        |  W  |
| ✊ Fist (hold 1.5s)   | Brake → Reverse   |  S  |
| V Sign                | Drift             |  S  |
| Point Index           | Nitro Boost       |  N  |
| 👈👉 Hand Left/Right  | Steer             | A/D |

## Quick Start

### Prerequisites

- Python 3.8 to 3.11
- Webcam
- Racing game with WASD+N controls

### Installation
```bash
# Clone repository
git clone https://github.com/arush-3009/hand-gesture-game-control.git
cd hand-gesture-game-control

# Install dependencies
pip install -r requirements.txt

# Run application
python main.py
```

### First Use

1. Position yourself in front of webcam
2. Run `python main.py`
3. Wait for window to open
4. Launch your racing game
5. Show gestures to control!

Press `q` to quit.

## ⚙️ Configuration

Edit `config.yml` to customize:
```yaml
# Steering sensitivity (0.0 = left edge, 1.0 = right edge)
gestures:
  left_threshold: 0.4   # Hand left of 40% = steer left
  right_threshold: 0.6  # Hand right of 60% = steer right
  smoothing_threshold: 3  # Higher = more stable, lower = more responsive

# Display mode
display:
  mode: "webcam"  # or "info_only" for better performance
```

## Display Modes

### Webcam Mode
- See your hand and camera feed
- Gesture labels overlaid
- Best for learning gestures

### Info Only Mode (Recommended for Gaming)
- Black background with minimal info
- Lower latency (~10-15ms faster)
- Steering visualization
- Best for competitive play

## Supported Games

Tested with:
- ✅ Asphalt 8: Airborne
- ✅ Asphalt 9: Legends
- ✅ Any game using WASD + N controls

## Architecture
```
Project/
├── src/
│   ├── camera.py          # Webcam management
│   ├── config.py          # Configuration loader
│   ├── display.py         # Visual feedback system
│   ├── gestures.py        # Gesture detection algorithms
│   ├── keyboard_input.py  # Keyboard control + smoothing
│   ├── tracking.py        # MediaPipe hand tracking
│   └── game_control.py    # Main game loop coordinator
├── main.py                # Entry point
└── config.yml             # Settings
```

## How It Works

1. **MediaPipe** detects 21 hand landmarks per frame
2. **Gesture algorithms** analyze landmark positions and ratios
3. **Smoothing filter** prevents flicker from brief detection failures
4. **Keyboard controller** simulates keypresses via pynput
5. **Display system** provides real-time visual feedback

### Gesture Detection Example
```python
# Open hand detection: All fingertips far from wrist
fingertip_distances = [tip - wrist for tip in fingertips]
is_open = all(distance > threshold * hand_size)
```

## Technical Highlights

- **Object-Oriented Design** - Clean separation of concerns
- **Config-Driven** - YAML configuration with safe defaults
- **Resource Management** - Proper cleanup with context managers
- **Error Handling** - Graceful failure with user feedback
- **Performance** - 30+ FPS with minimal CPU usage

## Troubleshooting

**Gestures not detected?**
- Ensure good lighting
- Keep hand 1-2 feet from camera
- Avoid cluttered background

**Controls laggy?**
- Switch to `info_only` mode in config
- Close other apps using webcam
- Reduce `smoothing_threshold` to 2

**Keys not registering in game?**
- Run as administrator (Windows)
- Check game is focused
- Verify game uses WASD+N controls

## Future Enhancements

- [ ] Two-hand mode (separate steering/controls)
- [ ] Machine learning for custom gestures
- [ ] Multi-game profiles
- [ ] Gesture calibration wizard
- [ ] Performance statistics dashboard


## License

MIT License - see [LICENSE](LICENSE) file for details
