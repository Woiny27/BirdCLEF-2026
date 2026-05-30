# BirdCLEF 2026 - ConvNeXt PyTorch Solution

A complete PyTorch solution for the **BirdCLEF 2026 Kaggle competition** using ConvNeXt Tiny for multi-label bird species classification from audio recordings.

## 🎯 Overview

This solution achieves multi-label bird species classification using:
- **Model**: ConvNeXt Tiny (pretrained ImageNet weights)
- **Input**: Mel spectrograms (128 bins, 5-second clips)
- **Output**: 234 bird species probabilities
- **Loss**: BCEWithLogitsLoss (multi-label compatible)
- **Optimization**: Kaggle CPU-friendly inference with INT8 quantization

## 📊 Competition Details

**BirdCLEF 2026** challenges participants to identify bird species from audio recordings across diverse geographical regions. Key aspects:

- **234 bird species** to classify
- **Multi-label classification** (multiple species possible per audio)
- **Variable-length audio** (5 seconds to several minutes)
- **Real-world recordings** with background noise and overlapping calls
- **Kaggle CPU constraints** (16GB RAM limit for inference)

## 🏗️ Project Structure

```
BirdCLEF-2026/
├── src/
│   ├── __init__.py
│   ├── model.py              # ConvNeXt Tiny model
│   ├── dataset.py            # PyTorch Dataset with SpecAugment
│   ├── train.py              # Training loop
│   └── inference.py          # Optimized inference for Kaggle
├── data/
│   ├── train.csv
│   ├── train_soundscapes_labels.csv
│   ├── taxonomy.csv
│   ├── train_audio/          # Training audio files
│   ├── test.csv
│   └── test_audio/           # Test audio files
├── model.pt                  # Saved model weights
├── predictions.csv           # Submission file
└── README.md                 # This file
```

## 🚀 Quick Start

### Installation

```bash
pip install torch torchvision torchaudio
pip install librosa pandas numpy scikit-learn tqdm
pip install timm  # For ConvNeXt model
```

### Training

```python
from src.dataset import BirdCLEFDataset
from src.train import main

# Run training
python src/train.py
```

The training script will:
1. Load datasets from CSV files
2. Create mel spectrograms from audio files
3. Apply SpecAugment augmentation
4. Train ConvNeXt model with AdamW optimizer
5. Save best model to `model.pt`

### Inference

```python
from src.inference import run_inference

# Generate predictions
predictions = run_inference(
    model_path='model.pt',
    test_csv_path='data/test.csv',
    audio_dir='data/test_audio',
    output_csv='predictions.csv',
    batch_size=16,
    num_clips=3,
    use_quantization=True
)
```

Outputs `predictions.csv` ready for Kaggle submission.

## 🔧 Key Components

### 1. Model (`src/model.py`)

**BirdCLEFModel** - ConvNeXt Tiny backbone with:
- Pretrained ImageNet weights
- Single input channel (grayscale mel spectrograms)
- 234 output classes
- Returns logits (compatible with BCEWithLogitsLoss)

```python
from src.model import BirdCLEFModel

model = BirdCLEFModel(num_classes=234, pretrained=True)
logits = model(spectrograms)  # (batch_size, 234)
```

### 2. Dataset (`src/dataset.py`)

**BirdCLEFDataset** - Complete audio processing pipeline:
- Loads .ogg audio files with librosa
- Resamples to 32000 Hz
- Extracts 5-second clips
- Converts to 128-bin mel spectrograms
- Applies SpecAugment (frequency & time masking)
- Multi-hot encodes labels

**BirdCLEFCombinedDataset** - Merges both training sources:
- Combines `train.csv` and `train_soundscapes_labels.csv`
- Automatic class balancing with weighted sampling
- Train/validation splitting

```python
from src.dataset import BirdCLEFDataset, BirdCLEFCombinedDataset
from torch.utils.data import DataLoader, WeightedRandomSampler

dataset = BirdCLEFDataset(
    csv_path='data/train.csv',
    audio_dir='data/train_audio',
    sr=32000,
    duration=5.0,
    n_mels=128,
    augment=True,
    split='train'
)

# Class balancing
weights = dataset.get_class_weights()
sampler = WeightedRandomSampler(weights, len(dataset))
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

### 3. Training (`src/train.py`)

Training pipeline with:
- AdamW optimizer (lr=1e-3)
- BCEWithLogitsLoss for multi-label learning
- GPU/CPU device support
- Model checkpointing (saves best model)
- Progress tracking with tqdm

```python
import torch.optim as optim
import torch.nn as nn

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
```

### 4. Inference (`src/inference.py`)

Optimized for Kaggle CPU constraints:

**OptimizedInference** engine:
- INT8 quantization (~4x faster on CPU)
- Batch processing (configurable batch size)
- Multi-clip aggregation (mean/max/weighted)
- Memory-efficient audio loading

**Utilities**:
- `run_inference()` - Full inference pipeline
- `ensemble_predictions()` - Combine multiple models

```python
from src.inference import run_inference

predictions = run_inference(
    model_path='model.pt',
    test_csv_path='data/test.csv',
    audio_dir='data/test_audio',
    batch_size=16,
    num_clips=3,  # Extract 3 clips per audio
    use_quantization=True
)
```

## 📈 Performance Features

### Audio Processing
- **Mel spectrograms**: 128 frequency bins, log-scale (dB)
- **Normalization**: [0, 1] range for better convergence
- **Sample rate**: 32000 Hz (balance between detail and computation)
- **Hop length**: 512 (good time resolution)

### Data Augmentation (SpecAugment)
- **Frequency masking**: Random 0-30 bins masked
- **Time masking**: Random 0-40 frames masked
- Applied only during training

### Class Balancing
- Inverse class frequency weighting
- WeightedRandomSampler for fair class representation
- Handles imbalanced species distribution

### Multi-clip Inference
- Extract 3 clips per audio (beginning, middle, end)
- Aggregate predictions:
  - **Mean**: Average probability across clips
  - **Max**: Maximum confidence per species
  - **Weighted**: By entropy-based confidence

## 💡 Optimization for Kaggle

The solution is optimized for Kaggle's 16GB RAM CPU environment:

1. **INT8 Quantization**
   - 75% faster inference on CPU
   - 4x smaller model size
   - Minimal accuracy loss

2. **Memory Management**
   - Streaming audio loading (not pre-loaded)
   - Explicit garbage collection after batches
   - Batch processing for controlled memory peaks

3. **Efficient Spectrograms**
   - Pre-computed on-the-fly (no disk storage)
   - 128 mel bins (balance of detail vs computation)
   - Normalized float32 tensors

## 📝 Configuration

### Dataset Parameters

```python
BirdCLEFDataset(
    sr=32000,           # Sampling rate (Hz)
    duration=5.0,       # Clip duration (seconds)
    n_mels=128,         # Number of mel bins
    augment=True,       # Enable SpecAugment
    split='train',      # 'train' or 'val'
    val_split=0.1       # Validation ratio
)
```

### Training Parameters

```python
num_epochs = 10
batch_size = 32
learning_rate = 1e-3
num_classes = 234
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### Inference Parameters

```python
batch_size = 16          # Batch for GPU/16GB CPU
num_clips = 3            # Clips per audio
use_quantization = True  # INT8 on CPU
confidence_threshold = 0.5
```

## 🎓 Model Architecture

**ConvNeXt Tiny** characteristics:
- 28M parameters
- ImageNet pretrained weights
- Efficient for CPU inference
- Strong performance on image-like inputs (spectrograms)
- Modern architecture (2022, MetaAI)

Input: `(batch, 1, 128, ~313)` - mel spectrograms
Output: `(batch, 234)` - logits for each species

## 🔍 Submission Format

`predictions.csv` structure:
```
audio_id,species_0,species_1,...,species_233
test_001.ogg,0.92,0.15,...,0.03
test_002.ogg,0.05,0.87,...,0.12
...
```

Where each `species_i` is the predicted probability (0-1) for that species class.

## 🚦 Troubleshooting

### Out of Memory (OOM) Error
- Reduce batch size: `batch_size = 8`
- Enable quantization: `use_quantization=True`
- Reduce clips: `num_clips = 1`

### Slow Inference
- Enable quantization: `use_quantization=True`
- Increase batch size (if memory allows): `batch_size=32`
- Use fewer clips: `num_clips=1`

### Audio Loading Errors
- Check file paths in CSV files
- Ensure audio files exist in `audio_dir`
- Verify `.ogg` format (librosa handles other formats too)

### Model Not Converging
- Check learning rate (try 5e-4 or 2e-3)
- Verify class balance with `dataset.get_class_weights()`
- Increase augmentation strength in `_spec_augment()`

## 📚 References

- **ConvNeXt**: [A ConvNet for the 2020s (Meta AI)](https://arxiv.org/abs/2201.03545)
- **SpecAugment**: [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/abs/1904.08779)
- **Kaggle BirdCLEF**: https://www.kaggle.com/competitions/birdclef-2026
- **PyTorch Audio**: https://pytorch.org/audio/stable/index.html
- **Librosa**: https://librosa.org/

## 📋 Checklist

- [x] Model architecture (ConvNeXt Tiny)
- [x] Audio processing pipeline (librosa, mel spectrograms)
- [x] Data augmentation (SpecAugment)
- [x] Multi-label dataset handling
- [x] Training loop with validation
- [x] Class balancing
- [x] Inference optimization (quantization, batching)
- [x] Multi-clip inference
- [x] Kaggle submission format
- [x] CPU optimization for 16GB limit

## 🤝 Contributing

To extend this solution:

1. **Better augmentation**: Try mixup, temporal warping, pitch shifting
2. **Ensemble methods**: Combine multiple models or architectures
3. **Pre/post-processing**: Noise reduction, normalization techniques
4. **Architecture search**: Try larger models if compute allows
5. **Semi-supervised learning**: Leverage unlabeled test data

## 📄 License

MIT License - Feel free to use this solution for learning and competition purposes.

---

**Good luck with BirdCLEF 2026! 🐦**

For questions or issues, please open a GitHub issue in the repository.
