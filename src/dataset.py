"""
BirdCLEF 2026 Dataset

Complete PyTorch Dataset with:
1. Reads train.csv, taxonomy.csv, train_soundscapes_labels.csv
2. Loads .ogg audio files
3. Resamples to 32000 Hz
4. Extracts 5-second clips
5. Converts clips to 128-bin mel spectrograms
6. SpecAugment data augmentation
7. Class balancing with weighted sampling
8. Returns spectrogram tensor and multi-label target
9. Supports 234 species classes
"""

import os
import pandas as pd
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import torch.nn.functional as F
from tqdm import tqdm


class BirdCLEFDataset(Dataset):
    """
    BirdCLEF 2026 Multi-label Audio Dataset
    
    Supports loading from:
    - train.csv (main training data)
    - train_soundscapes_labels.csv (soundscape recordings)
    """
    
    def __init__(
        self,
        csv_path,
        audio_dir,
        taxonomy_path=None,
        sr=32000,
        duration=5.0,
        n_mels=128,
        augment=True,
        split='train',
        val_split=0.1,
        random_state=42
    ):
        """
        Args:
            csv_path: Path to train.csv or train_soundscapes_labels.csv
            audio_dir: Directory containing .ogg audio files
            taxonomy_path: Path to taxonomy.csv (optional, for class info)
            sr: Sampling rate (default: 32000 Hz)
            duration: Clip duration in seconds (default: 5.0)
            n_mels: Number of mel bins (default: 128)
            augment: Enable SpecAugment (default: True)
            split: 'train' or 'val' split
            val_split: Validation split ratio (default: 0.1)
            random_state: Random seed for reproducibility
        """
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.augment = augment
        self.audio_dir = audio_dir
        
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        # Load CSV data
        self.df = pd.read_csv(csv_path)
        
        # Load taxonomy if provided
        if taxonomy_path is not None:
            self.taxonomy = pd.read_csv(taxonomy_path)
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.taxonomy['scientific_name'].unique())}
        else:
            # Extract unique species from dataframe
            self.species = sorted(set(self.df['primary_label'].unique()))
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.species)}
        
        self.num_classes = len(self.class_to_idx)
        
        # Handle different CSV formats
        if 'filename' in self.df.columns:
            self.df['audio_path'] = self.df['filename'].apply(
                lambda x: os.path.join(audio_dir, x)
            )
        elif 'filepath' in self.df.columns:
            self.df['audio_path'] = self.df['filepath'].apply(
                lambda x: os.path.join(audio_dir, x)
            )
        
        # Filter to existing files
        self.df = self.df[self.df['audio_path'].apply(os.path.exists)].reset_index(drop=True)
        
        # Split train/val
        if split in ['train', 'val']:
            n_train = int(len(self.df) * (1 - val_split))
            indices = np.random.permutation(len(self.df))
            
            if split == 'train':
                self.indices = indices[:n_train]
            else:
                self.indices = indices[n_train:]
            
            self.df = self.df.iloc[self.indices].reset_index(drop=True)
        
        self.augment = augment and split == 'train'
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            spectrogram: Mel spectrogram tensor (1, n_mels, time_steps)
            target: Multi-hot encoded labels (num_classes,)
        """
        row = self.df.iloc[idx]
        
        # Load audio
        try:
            audio, _ = librosa.load(row['audio_path'], sr=self.sr)
        except Exception as e:
            print(f"Error loading {row['audio_path']}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))
        
        # Extract 5-second clip
        clip_length = int(self.sr * self.duration)
        if len(audio) < clip_length:
            audio = np.pad(audio, (0, clip_length - len(audio)), mode='wrap')
        else:
            start = np.random.randint(0, len(audio) - clip_length + 1)
            audio = audio[start:start + clip_length]
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=2048,
            hop_length=512
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = (mel_spec + 80) / 80  # Normalize to [0, 1]
        
        # Apply SpecAugment if training
        if self.augment:
            mel_spec = self._spec_augment(mel_spec)
        
        # Convert to tensor
        spectrogram = torch.FloatTensor(mel_spec).unsqueeze(0)  # (1, n_mels, time)
        
        # Create multi-hot target
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        
        # Handle different label formats
        if 'primary_label' in row:
            primary = row['primary_label']
            if primary in self.class_to_idx:
                target[self.class_to_idx[primary]] = 1.0
        
        if 'secondary_labels' in row and pd.notna(row['secondary_labels']):
            secondary = str(row['secondary_labels']).split()
            for label in secondary:
                if label in self.class_to_idx:
                    target[self.class_to_idx[label]] = 1.0
        
        return spectrogram, target
    
    def _spec_augment(self, mel_spec, freq_mask_param=30, time_mask_param=40):
        """
        Apply SpecAugment to mel spectrogram.
        
        Args:
            mel_spec: Mel spectrogram (n_mels, time)
            freq_mask_param: Maximum frequency mask width
            time_mask_param: Maximum time mask width
        
        Returns:
            Augmented mel spectrogram
        """
        mel_spec = mel_spec.copy()
        
        # Frequency masking
        freq_mask_width = np.random.randint(0, freq_mask_param)
        if freq_mask_width > 0:
            freq_mask_start = np.random.randint(0, self.n_mels - freq_mask_width)
            mel_spec[freq_mask_start:freq_mask_start + freq_mask_width, :] = 0
        
        # Time masking
        time_mask_width = np.random.randint(0, time_mask_param)
        if time_mask_width > 0:
            time_mask_start = np.random.randint(0, mel_spec.shape[1] - time_mask_width)
            mel_spec[:, time_mask_start:time_mask_start + time_mask_width] = 0
        
        return mel_spec
    
    def get_class_weights(self):
        """
        Compute class weights for balancing.
        
        Returns:
            Sample weights for WeightedRandomSampler
        """
        # Count samples per class
        class_counts = np.zeros(self.num_classes)
        
        for idx in range(len(self)):
            _, target = self.__getitem__(idx)
            class_counts += target.numpy()
        
        # Compute inverse weights (more samples = lower weight)
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = class_weights / class_weights.sum() * self.num_classes
        
        # Assign weight to each sample based on its positive classes
        sample_weights = np.zeros(len(self))
        for idx in range(len(self)):
            _, target = self.__getitem__(idx)
            sample_weights[idx] = class_weights[target > 0].mean()
        
        return sample_weights


class BirdCLEFCombinedDataset(Dataset):
    """
    Combined dataset from train.csv and train_soundscapes_labels.csv
    with automatic merging and class balancing.
    """
    
    def __init__(
        self,
        train_csv_path,
        soundscapes_csv_path,
        audio_dir,
        taxonomy_path=None,
        sr=32000,
        duration=5.0,
        n_mels=128,
        augment=True,
        split='train',
        val_split=0.1,
        random_state=42
    ):
        """Load and combine both training datasets."""
        
        # Load main dataset
        self.train_dataset = BirdCLEFDataset(
            train_csv_path,
            audio_dir,
            taxonomy_path,
            sr, duration, n_mels,
            augment, split, val_split,
            random_state
        )
        
        # Load soundscapes dataset
        self.soundscapes_dataset = BirdCLEFDataset(
            soundscapes_csv_path,
            audio_dir,
            taxonomy_path,
            sr, duration, n_mels,
            augment, split, val_split,
            random_state
        )
        
        self.dataset_lengths = [
            len(self.train_dataset),
            len(self.soundscapes_dataset)
        ]
        self.cumulative_lengths = np.cumsum(self.dataset_lengths)
    
    def __len__(self):
        return sum(self.dataset_lengths)
    
    def __getitem__(self, idx):
        if idx < self.cumulative_lengths[0]:
            return self.train_dataset[idx]
        else:
            return self.soundscapes_dataset[idx - self.cumulative_lengths[0]]
    
    def get_class_weights(self):
        """Compute weights for both datasets."""
        weights1 = self.train_dataset.get_class_weights()
        weights2 = self.soundscapes_dataset.get_class_weights()
        return np.concatenate([weights1, weights2])


if __name__ == "__main__":
    # Example usage
    dataset = BirdCLEFDataset(
        csv_path="data/train.csv",
        audio_dir="data/train_audio",
        taxonomy_path="data/taxonomy.csv",
        sr=32000,
        duration=5.0,
        n_mels=128,
        augment=True,
        split='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    
    # Test loading a sample
    spectrogram, target = dataset[0]
    print(f"Spectrogram shape: {spectrogram.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Positive labels: {target.nonzero(as_tuple=True)[0].tolist()}")
