"""
Optimized inference for Kaggle CPU notebook limits.

Requirements:
- Run under Kaggle 16GB RAM limit
- Batch processing for efficiency
- Model quantization (INT8)
- Memory-efficient audio loading
- Fast prediction on CPU
- Submit predictions.csv
"""

import os
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import gc
from tqdm import tqdm

from src.model import BirdCLEFModel


class TestAudioDataset(Dataset):
    """Memory-efficient test audio dataset for inference."""
    
    def __init__(
        self,
        test_csv_path,
        audio_dir,
        sr=32000,
        duration=5.0,
        n_mels=128,
        num_clips=3
    ):
        """
        Args:
            test_csv_path: Path to test.csv or test_soundscapes.csv
            audio_dir: Directory containing .ogg audio files
            sr: Sampling rate
            duration: Clip duration in seconds
            n_mels: Number of mel bins
            num_clips: Number of clips to extract per audio
        """
        self.df = pd.read_csv(test_csv_path)
        self.audio_dir = audio_dir
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.num_clips = num_clips
        
        # Build audio paths
        if 'filename' in self.df.columns:
            self.df['audio_path'] = self.df['filename'].apply(
                lambda x: os.path.join(audio_dir, x)
            )
        
        # Filter existing files
        self.df = self.df[self.df['audio_path'].apply(os.path.exists)].reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Returns all clips for an audio file."""
        row = self.df.iloc[idx]
        audio_id = row.get('audio_id', row.get('filename', f'audio_{idx}'))
        
        try:
            # Load audio
            audio, _ = librosa.load(row['audio_path'], sr=self.sr)
        except Exception as e:
            print(f"Error loading {row['audio_path']}: {e}")
            return audio_id, []
        
        clip_length = int(self.sr * self.duration)
        clips = []
        
        # Extract multiple clips from different positions
        if len(audio) < clip_length:
            audio = np.pad(audio, (0, clip_length - len(audio)), mode='wrap')
            clips.append(self._audio_to_spectrogram(audio))
        else:
            positions = np.linspace(0, len(audio) - clip_length, self.num_clips, dtype=int)
            for pos in positions:
                clip = audio[pos:pos + clip_length]
                clips.append(self._audio_to_spectrogram(clip))
        
        # Clear memory
        del audio
        gc.collect()
        
        return audio_id, clips
    
    def _audio_to_spectrogram(self, audio):
        """Convert audio to mel spectrogram."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=2048,
            hop_length=512
        )
        
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = (mel_spec + 80) / 80  # Normalize to [0, 1]
        
        return torch.FloatTensor(mel_spec).unsqueeze(0)


class OptimizedInference:
    """Optimized inference engine for Kaggle CPU."""
    
    def __init__(
        self,
        model_path,
        num_classes=234,
        batch_size=16,
        use_quantization=True,
        device='cpu'
    ):
        """
        Args:
            model_path: Path to saved model weights
            num_classes: Number of output classes
            batch_size: Batch size for inference
            use_quantization: Use INT8 quantization
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.num_classes = num_classes
        
        # Create and load model
        self.model = BirdCLEFModel(num_classes=num_classes, pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Apply quantization if on CPU
        if use_quantization and device == 'cpu':
            self.model = self._quantize_model(self.model)
        
        print(f"Model loaded on {self.device}")
    
    def _quantize_model(self, model):
        """Apply INT8 quantization for faster CPU inference."""
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8
            )
            print("Model quantized to INT8")
            return quantized_model
        except Exception as e:
            print(f"Quantization failed: {e}. Using original model.")
            return model
    
    @torch.no_grad()
    def predict_batch(self, spectrograms):
        """
        Predict on a batch of spectrograms.
        
        Args:
            spectrograms: List of (1, n_mels, time) tensors
        
        Returns:
            Predictions of shape (batch_size, num_classes)
        """
        # Stack and move to device
        batch = torch.stack(spectrograms).to(self.device)
        
        # Forward pass
        logits = self.model(batch)
        
        # Convert to probabilities using sigmoid
        probs = torch.sigmoid(logits)
        
        return probs.cpu().numpy()
    
    def predict_multi_clip(self, clips, method='mean'):
        """
        Predict on multiple clips from same audio.
        
        Args:
            clips: List of spectrograms
            method: 'mean', 'max', or 'weighted' aggregation
        
        Returns:
            Aggregated prediction
        """
        if not clips:
            return np.zeros(self.num_classes)
        
        # Batch predict all clips
        predictions = self.predict_batch(clips)
        
        # Aggregate
        if method == 'mean':
            return predictions.mean(axis=0)
        elif method == 'max':
            return predictions.max(axis=0)
        elif method == 'weighted':
            # Weight by confidence (entropy)
            weights = 1.0 / (1.0 + entropy(predictions))
            weights = weights / weights.sum()
            return (predictions * weights[:, None]).sum(axis=0)
        else:
            return predictions.mean(axis=0)


def entropy(predictions):
    """Compute entropy for confidence weighting."""
    eps = 1e-10
    return -np.sum(
        predictions * np.log(predictions + eps) +
        (1 - predictions) * np.log(1 - predictions + eps),
        axis=1
    )


def run_inference(
    model_path,
    test_csv_path,
    audio_dir,
    output_csv='predictions.csv',
    batch_size=16,
    num_clips=3,
    confidence_threshold=0.5,
    use_quantization=True
):
    """
    Run inference on test set.
    
    Args:
        model_path: Path to trained model
        test_csv_path: Path to test.csv
        audio_dir: Directory with test audio files
        output_csv: Output predictions filename
        batch_size: Batch size for inference
        num_clips: Number of clips per audio
        confidence_threshold: Threshold for predictions
        use_quantization: Use INT8 quantization
    """
    
    # Create dataset
    dataset = TestAudioDataset(
        test_csv_path,
        audio_dir,
        num_clips=num_clips
    )
    
    # Initialize inference engine
    inference = OptimizedInference(
        model_path,
        batch_size=batch_size,
        use_quantization=use_quantization,
        device='cpu'
    )
    
    # Species list for submission
    species = sorted([f'species_{i}' for i in range(inference.num_classes)])
    
    # Run inference
    results = []
    
    for audio_id, clips in tqdm(dataset, desc="Inferencing"):
        if not clips:
            # No clips (file error) - predict zeros
            pred = np.zeros(inference.num_classes)
        else:
            # Multi-clip aggregation
            pred = inference.predict_multi_clip(clips, method='mean')
        
        # Apply threshold and create submission row
        row = {'audio_id': audio_id}
        for i, species_name in enumerate(species):
            row[species_name] = float(pred[i])
        
        results.append(row)
        
        # Periodically clear memory
        if len(results) % 100 == 0:
            gc.collect()
    
    # Save predictions
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    
    return submission_df


def ensemble_predictions(prediction_files, output_csv='ensemble_predictions.csv'):
    """
    Ensemble multiple prediction files.
    
    Args:
        prediction_files: List of CSV file paths
        output_csv: Output ensemble predictions
    
    Returns:
        Ensemble predictions DataFrame
    """
    predictions = []
    
    for pred_file in prediction_files:
        df = pd.read_csv(pred_file)
        predictions.append(df)
    
    # Average predictions
    ensemble_df = predictions[0].copy()
    
    for col in ensemble_df.columns:
        if col != 'audio_id':
            ensemble_df[col] = np.mean(
                [pred[col].values for pred in predictions],
                axis=0
            )
    
    ensemble_df.to_csv(output_csv, index=False)
    print(f"Ensemble predictions saved to {output_csv}")
    
    return ensemble_df


if __name__ == "__main__":
    # Example usage
    predictions = run_inference(
        model_path='model.pt',
        test_csv_path='data/test.csv',
        audio_dir='data/test_audio',
        output_csv='predictions.csv',
        batch_size=16,
        num_clips=3,
        use_quantization=True
    )
    
    print(predictions.head())
