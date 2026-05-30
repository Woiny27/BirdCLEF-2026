"""
BirdCLEF 2026 Dataset

Create a PyTorch Dataset that:

1. Reads train.csv
2. Loads .ogg audio files
3. Resamples to 32000 Hz
4. Extracts 5-second clips
5. Converts clips to 128-bin mel spectrograms
6. Returns spectrogram tensor and multi-label target
7. Supports 234 species classes
"""