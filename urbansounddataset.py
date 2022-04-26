import pandas as pd
import torchaudio
import os
import torch

from torch.utils.data import Dataset


class UrbanSoundDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        audio_dir,
        transformation,
        target_sample_rate,
        num_samples,
        device,
    ):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)  # Use GPU
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        # get tensor of audio and the sample rate
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(
            self.device
        )  # Signal registered onto device, able to use GPU
        signal = self._clean_signal(signal, sr)
        signal = self.transformation(signal)
        return signal, label

    # Performs resampling, demixing and cutting/padding
    def _clean_signal(self, signal, sr):
        # signal.shape -> (num_channels, samples)
        signal = self._resample_if_necessary(signal, sr)
        # Signals might have different sample rates
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        return signal

    def _cut_if_necessary(self, signal):
        # signal is tensor of (1, num_samples)
        # if length of signal more than num_samples
        # cut until num_samples
        if signal.shape[1] > self.num_samples:
            signal = signal[:, : self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        if signal.shape[1] < self.num_samples:
            num_missing_samples = self.num_samples - signal.shape[1]
            # 1st argument represents amount to prepend padding
            # 2nd argument represents amount to append padding
            last_dim_padding = (0, num_missing_samples)
            # pad function starts padding from last dimension
            # eg if pad(1, 1, 2, 2), pad last dim by (1, 1) and 2nd to last by (2, 2)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    # Convert original sampling rate to a specified sampling rate
    # single underscore methods for internal use
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(
                self.device
            )  # Use GPU
            signal = resampler(signal)
        return signal

    # Convert to mono channel
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"  # see csv for locations
        file_name = self.annotations.iloc[index, 0]
        path = os.path.join(self.audio_dir, fold, file_name)
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]  # see csv for locations


if __name__ == "__main__":
    ANNOTATIONS_FILE = "UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "UrbanSound8K/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    # Checks if GPU (cuda) is available and use it. If not, use CPU
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")

    # Create callable object -> mel_spectrogram can be directly applied to the signal
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
    )

    usd = UrbanSoundDataset(
        ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device
    )
