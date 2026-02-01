# Video Documentation

## Overview

This document describes the video files generated to demonstrate the bipedal robot environment and control policies.

## Generated Videos

### Demo Videos (`results/demo/videos/`)

Three demonstration videos showing different control policies:

1. **episode_0_random.mp4** (147 KB)
   - **Policy**: Random actions (uniform distribution between -1 and 1)
   - **Purpose**: Baseline showing robot behavior without any learned control
   - **Duration**: 200 steps (~7.7 seconds at 30 FPS)
   - **Reward**: 75.23
   - **Description**: Robot with completely random joint movements

2. **episode_1_forward_bias.mp4** (146 KB)
   - **Policy**: Random actions with slight forward bias
   - **Purpose**: Demonstrates simple heuristic control
   - **Duration**: 200 steps (~7.7 seconds at 30 FPS)
   - **Reward**: 74.18
   - **Description**: Random movements with added forward push on primary joint

3. **episode_2_standing.mp4** (147 KB)
   - **Policy**: Stabilization policy attempting to maintain upright posture
   - **Purpose**: Shows simple feedback control
   - **Duration**: 200 steps (~7.7 seconds at 30 FPS)
   - **Reward**: 72.51
   - **Description**: Robot attempting to counteract tilt using orientation feedback

### Algorithm-Specific Videos

Sample videos have been copied to each algorithm's directory for reference:

- **results/sac/videos/episode_0.mp4** (146 KB)
- **results/dreamer/videos/episode_0.mp4** (146 KB)
- **results/hierarchical/videos/episode_0.mp4** (146 KB)

These are copies of the forward bias policy video, demonstrating the expected output format for trained policies.

## Video Specifications

- **Format**: MP4 (H.264)
- **Frame Rate**: 30 FPS
- **Resolution**: 320x240 pixels (RGB)
- **Encoding**: FFmpeg via imageio-ffmpeg
- **Average File Size**: ~146 KB per 200 frames

## Environment Details

### Robot Model
- **Type**: Humanoid biped from PyBullet
- **URDF**: `humanoid/humanoid.urdf` (PyBullet data)
- **Controllable Joints**: 4 (determined from URDF)
  - Joints include hip, ankle, shoulder actuators
  - Each joint receives continuous action values in [-1, 1]

### Rendering
- **Mode**: RGB array (headless rendering)
- **Camera**: Fixed third-person view
- **Background**: Ground plane with grid
- **Lighting**: Default PyBullet lighting

## How Videos Were Generated

### Generation Script
Videos were created using `generate_sample_videos.py`:

```bash
python generate_sample_videos.py
```

### Process
1. Create PyBullet environment in RGB array mode (no GUI)
2. Reset environment to initial state
3. For each timestep:
   - Compute action from policy function
   - Apply action to robot
   - Render RGB frame (320x240)
   - Store frame in memory
4. Save all frames as MP4 using imageio and FFmpeg

### Policy Functions

**Random Policy**:
```python
def random_policy(obs, step):
    return np.random.uniform(-1, 1, size=action_dim)
```

**Forward Bias Policy**:
```python
def forward_bias_policy(obs, step):
    action = np.random.uniform(-0.5, 0.5, size=action_dim)
    action[0] = np.clip(0.3 + np.random.normal(0, 0.1), -1, 1)
    return action
```

**Standing Policy**:
```python
def standing_policy(obs, step):
    action = np.zeros(action_dim)
    action[0] = -obs[2] * 0.5  # Pitch correction
    action[1] = -obs[3] * 0.5  # Roll correction
    return np.clip(action, -1, 1)
```

## Expected Trained Policy Videos

When running full training with the three algorithms, you will generate:

### SAC (Model-Free)
- **Location**: `results/sac/videos/`
- **Expected Episodes**: 10 evaluation videos
- **Expected Behavior**: Stable forward walking gait
- **Training Time**: ~10 hours (1M steps)

### Dreamer (World Model)
- **Location**: `results/dreamer/videos/`
- **Expected Episodes**: 10 evaluation videos
- **Expected Behavior**: Sample-efficient learning of forward locomotion
- **Training Time**: ~5 hours (500k steps)

### Hierarchical
- **Location**: `results/hierarchical/videos/`
- **Expected Episodes**: 10 evaluation videos
- **Expected Behavior**: Diverse skills (walking, turning, recovery)
- **Training Time**: ~10 hours (1M steps)

## Viewing Videos

### Command Line
```bash
# Linux with VLC
vlc results/demo/videos/episode_0_random.mp4

# Linux with ffplay
ffplay results/demo/videos/episode_0_random.mp4

# Mac
open results/demo/videos/episode_0_random.mp4
```

### Python
```python
import cv2

# Read video
cap = cv2.VideoCapture('results/demo/videos/episode_0_random.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Robot', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Jupyter Notebook
```python
from IPython.display import Video

Video('results/demo/videos/episode_0_random.mp4', width=640)
```

## Video Generation from Trained Checkpoints

To generate videos from your own trained policies:

```bash
python evaluate.py \
    --config configs/sac_config.yaml \
    --checkpoint results/sac/checkpoints/final_model.pt \
    --num_episodes 10 \
    --save_videos
```

This will:
1. Load the trained agent from checkpoint
2. Run evaluation episodes
3. Save videos to the configured video directory
4. Print evaluation statistics

## Troubleshooting

### Videos Won't Play
- Ensure FFmpeg is installed: `pip install imageio-ffmpeg`
- Check codec support in your video player
- Try converting: `ffmpeg -i input.mp4 -c:v libx264 output.mp4`

### Empty or Corrupted Videos
- Verify PyBullet rendering works: check `render_mode='rgb_array'`
- Ensure sufficient disk space
- Check write permissions on output directory

### Performance Issues
- Rendering adds overhead: ~26 FPS vs ~50+ FPS without rendering
- Use headless mode (`use_gui=False`) for faster generation
- Generate videos only during evaluation, not training

## Technical Notes

### Frame Collection
Frames are collected in memory as numpy arrays (RGB, uint8) during rollout:
```python
frame = env.render()  # Returns (H, W, 3) array
frames.append(frame)
```

### Video Encoding
Using imageio with FFmpeg backend:
```python
from src.utils.video import save_video

save_video(frames, 'output.mp4', fps=30)
```

Implementation uses imageio's `mimwrite` with appropriate codec settings for compatibility.

### Memory Considerations
- Each frame: ~230 KB (320x240x3 bytes)
- 200 frames: ~45 MB RAM
- 1000 frames: ~225 MB RAM

For long episodes, consider:
- Downsampling frames (save every Nth frame)
- Lower resolution rendering
- Streaming to disk instead of memory buffering

## Future Enhancements

Potential improvements for video generation:

1. **Multi-camera views**: Side, top, first-person perspectives
2. **Overlay information**: Reward, action values, internal states
3. **Comparison videos**: Side-by-side policy comparisons
4. **Slow motion**: Highlight interesting behaviors
5. **GIF generation**: For easy sharing and documentation
6. **Tensorboard integration**: Embed videos in training logs

## References

- PyBullet Documentation: https://pybullet.org/
- imageio Documentation: https://imageio.readthedocs.io/
- FFmpeg: https://ffmpeg.org/

## File Checksums

For verification purposes:

```bash
# Generate checksums
md5sum results/demo/videos/*.mp4
# or
sha256sum results/demo/videos/*.mp4
```

Generated on: 2026-02-01

## License

Videos are generated from PyBullet's built-in humanoid model and are provided for educational and research purposes under the same license as the codebase.
