import torch
import torchaudio
import numpy as np
from torch_audiomentations import Compose, OneOf
from torch_audiomentations import (
    PitchShift,
    Shift,
    LowPassFilter,
    AddColoredNoise,
    BandPassFilter,
    Gain,
    PeakNormalization
)

import torchaudio.transforms as T
import tempfile
import os
import random
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from types import SimpleNamespace
import torchaudio.functional as F


class AlgorithmicReverb(BaseWaveformTransform):
    """Applies algorithmic reverberation using torchaudio's effects."""
    def __init__(self, min_reverberance=20, max_reverberance=80,
                 min_damping=10, max_damping=70,
                 min_room_scale=50, max_room_scale=100,
                 p=0.5, output_type="tensor", sample_rate=16000):
        """Initializes the reverb transform with parameter ranges."""
        super().__init__(p=p, output_type=output_type, sample_rate=sample_rate)
        self.min_reverberance = min_reverberance
        self.max_reverberance = max_reverberance
        self.min_damping = min_damping
        self.max_damping = max_damping
        self.min_room_scale = min_room_scale
        self.max_room_scale = max_room_scale

    def randomize_parameters(self, samples, sample_rate, targets=None, target_rate=None):
        """Randomizes reverb parameters for each application."""
        super().randomize_parameters(samples, sample_rate, targets, target_rate)
        self.transform_parameters["reverberance"] = random.uniform(self.min_reverberance, self.max_reverberance)
        self.transform_parameters["damping"] = random.uniform(self.min_damping, self.max_damping)
        self.transform_parameters["room_scale"] = random.uniform(self.min_room_scale, self.max_room_scale)
        self.transform_parameters["wet_gain"] = random.uniform(-5, 5)

    def apply_transform(self, samples, sample_rate, targets=None, target_rate=None):
        """Applies the reverberation effect to the audio samples."""
        samples_2d = samples.squeeze(0)

        effects = [
            ["reverb",
             "-w",
             str(self.transform_parameters["reverberance"]),
             str(self.transform_parameters["damping"]),
             str(self.transform_parameters["room_scale"]),
             str(self.transform_parameters["wet_gain"])]
        ]

        reverbed_samples_2d, _ = F.apply_effects_tensor(
            tensor=samples_2d,
            sample_rate=sample_rate,
            effects=effects,
        )

        return reverbed_samples_2d.unsqueeze(0)


class MP3Compression(BaseWaveformTransform):
    """Simulates MP3 compression by saving and reloading the audio via ffmpeg."""
    def __init__(self, min_bitrate=64, max_bitrate=128, p=0.5, sample_rate=16000, output_type='tensor'):
        """Initializes the MP3 compression transform."""
        super().__init__(p=p, output_type="tensor", sample_rate=sample_rate)
        self.min_bitrate = min_bitrate
        self.max_bitrate = max_bitrate
        self.codec_simulator = AudioCodecSimulator(sample_rate)

    def randomize_parameters(self, samples, sample_rate, targets=None, target_rate=None):
        """Randomizes the target bitrate."""
        super().randomize_parameters(samples, sample_rate, targets, target_rate)
        self.transform_parameters["bitrate"] = random.randint(self.min_bitrate, self.max_bitrate)

    def apply_transform(self, samples, sample_rate, targets=None, target_rate=None):
        """Applies the MP3 compression simulation."""
        bitrate = self.transform_parameters["bitrate"]
        processed = self.codec_simulator.mp3_compression(samples, sample_rate, bitrate=bitrate)

        return SimpleNamespace(
            samples=processed,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


class TelephoneQuality(BaseWaveformTransform):
    """Simulates the audio quality of a telephone line."""
    def __init__(self, p=0.5, output_type="tensor", sample_rate=16000):
        """Initializes the telephone quality transform."""
        super().__init__(p=p, output_type=output_type, sample_rate=sample_rate)
        self.codec_simulator = AudioCodecSimulator(sample_rate)

    def apply_transform(self, samples, sample_rate, targets=None, target_rate=None):
        """Applies band-pass filtering, quantization, and noise."""
        processed = self.codec_simulator.telephone_quality(samples, sample_rate)

        return SimpleNamespace(
            samples=processed,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


class LowBitrateCompression(BaseWaveformTransform):
    """Simulates low-bitrate compression by downsampling and upsampling."""
    def __init__(self, min_bitrate=32, max_bitrate=64, p=0.5, output_type="tensor", sample_rate=16000):
        """Initializes the low-bitrate compression transform."""
        super().__init__(p=p, output_type=output_type, sample_rate=sample_rate)
        self.min_bitrate = min_bitrate
        self.max_bitrate = max_bitrate
        self.codec_simulator = AudioCodecSimulator(sample_rate)

    def randomize_parameters(self, samples, sample_rate, targets=None, target_rate=None):
        """Randomizes the target bitrate."""
        super().randomize_parameters(samples, sample_rate, targets, target_rate)
        self.transform_parameters["bitrate"] = random.randint(self.min_bitrate, self.max_bitrate)

    def apply_transform(self, samples, sample_rate, targets=None, target_rate=None):
        """Applies the resampling-based compression simulation."""
        bitrate = self.transform_parameters["bitrate"]
        processed = self.codec_simulator.low_bitrate_compression(samples, sample_rate, target_bitrate=bitrate)

        return SimpleNamespace(
            samples=processed,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


class AudioCodecSimulator:
    """A helper class to simulate various audio codec effects."""
    def __init__(self, sample_rate):
        """Initializes the simulator and its component transforms."""
        self.sample_rate = sample_rate

        self.bandpass = BandPassFilter(
            min_center_frequency=300,
            max_center_frequency=3400,
            p=1.0,
            output_type="tensor",
            sample_rate=self.sample_rate
        )
        self.noise = AddColoredNoise(
            min_snr_in_db=10,
            max_snr_in_db=20,
            p=1.0,
            output_type="tensor",
            sample_rate=self.sample_rate
        )

    def mp3_compression(self, waveform, sample_rate, bitrate=128):
        """Performs MP3 compression using ffmpeg or a fallback."""
        waveform_to_save = waveform.squeeze(0)
        try:
            import subprocess
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            return self.alternative_compression(waveform, bitrate)
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3:
                    torchaudio.save(tmp_wav.name, waveform_to_save, sample_rate)
                    os.system(
                        f'ffmpeg -i {tmp_wav.name} -codec:a libmp3lame -b:a {bitrate}k {tmp_mp3.name} -y -loglevel quiet')
                    if os.path.exists(tmp_mp3.name) and os.path.getsize(tmp_mp3.name) > 0:
                        compressed_waveform, _ = torchaudio.load(tmp_mp3.name)
                        os.unlink(tmp_wav.name)
                        os.unlink(tmp_mp3.name)
                        return compressed_waveform.unsqueeze(0)
                    else:
                        os.unlink(tmp_wav.name)
                        if os.path.exists(tmp_mp3.name):
                            os.unlink(tmp_mp3.name)
                        return self.alternative_compression(waveform, bitrate)
        except Exception:
            return self.alternative_compression(waveform, bitrate)

    def alternative_compression(self, waveform, bitrate):
        """A fallback compression method using quantization and noise."""
        quality_factor = min(bitrate / 320.0, 1.0)
        bits = int(8 + 8 * quality_factor)
        max_val = 2 ** (bits - 1) - 1
        compressed = torch.round(waveform * max_val) / max_val
        noise_level = 0.001 * (1 - quality_factor)
        noise = torch.randn_like(compressed) * noise_level
        return compressed + noise

    def telephone_quality(self, waveform, sample_rate):
        """Simulates telephone quality by filtering and adding noise."""
        quantized = torch.round(waveform * 127) / 127
        filtered = self.bandpass(samples=quantized, sample_rate=sample_rate)
        noisy = self.noise(samples=filtered, sample_rate=sample_rate)
        return noisy

    def low_bitrate_compression(self, waveform, sample_rate, target_bitrate=64):
        """Simulates low bitrate by resampling."""
        target_sr = min(sample_rate, 8000 + int(target_bitrate * 50))

        if target_sr < sample_rate:
            resample_down = T.Resample(sample_rate, target_sr)
            resample_up = T.Resample(target_sr, sample_rate)
            downsampled = resample_down(waveform)
            upsampled = resample_up(downsampled)

            if upsampled.shape[-1] > waveform.shape[-1]:
                return upsampled[..., :waveform.shape[-1]]
            elif upsampled.shape[-1] < waveform.shape[-1]:
                padding_amount = waveform.shape[-1] - upsampled.shape[-1]
                return torch.nn.functional.pad(upsampled, (0, padding_amount))
            else:
                return upsampled

        return waveform


class AudioAugmentor:
    """A comprehensive audio augmentation pipeline with dynamic parameters."""
    def __init__(self, sample_rate=16000, initial_p=0.5, initial_intensity=1.0):
        """Initializes the augmentor with dynamic probability and intensity."""
        self.sample_rate = sample_rate
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError(f"Received invalid sample_rate: {sample_rate}")

        self.p = initial_p
        self.intensity_coeff = initial_intensity
        self._create_transforms()

    def _create_transforms(self):
        """Builds the augmentation pipeline based on current parameters."""
        mp3_max_bitrate = max(32, int(128 / self.intensity_coeff))
        mp3_min_bitrate = max(32, int(64 / self.intensity_coeff))

        low_max_bitrate = max(16, int(64 / self.intensity_coeff))
        low_min_bitrate = max(16, int(32 / self.intensity_coeff))

        noise_min_snr = 3 / self.intensity_coeff
        noise_max_snr = 20 / self.intensity_coeff

        pitch_shift_semitones = 2 * self.intensity_coeff

        transforms = []
        transforms.append(
            OneOf([
                MP3Compression(min_bitrate=mp3_min_bitrate, max_bitrate=mp3_max_bitrate, p=1.0,
                               sample_rate=self.sample_rate),
                TelephoneQuality(p=1.0, sample_rate=self.sample_rate),
                LowBitrateCompression(min_bitrate=low_min_bitrate, max_bitrate=low_max_bitrate, p=1.0,
                                      sample_rate=self.sample_rate),
                AddColoredNoise(min_snr_in_db=noise_min_snr, max_snr_in_db=noise_max_snr, p=1.0,
                                sample_rate=self.sample_rate),
            ], p=0.4)
        )
        transforms.extend([
            Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.3, sample_rate=self.sample_rate),
            PitchShift(min_transpose_semitones=-pitch_shift_semitones, max_transpose_semitones=pitch_shift_semitones,
                       p=0.3, sample_rate=self.sample_rate),
            Shift(min_shift=-0.1, max_shift=0.1, p=0.3, sample_rate=self.sample_rate),
            LowPassFilter(min_cutoff_freq=2000, max_cutoff_freq=7500, p=0.3, sample_rate=self.sample_rate),
        ])
        transforms.append(PeakNormalization(p=1.0))
        self.augmentations = Compose(transforms=transforms, output_type="dict")

    def update_parameters(self, new_p, new_intensity_coeff):
        """Updates augmentation parameters and rebuilds the pipeline."""
        print(f"Updating augmentor: p={new_p:.3f}, intensity={new_intensity_coeff:.3f}")
        self.p = new_p
        self.intensity_coeff = new_intensity_coeff
        self._create_transforms()

    def __call__(self, audio_tensor):
        """Applies the full augmentation pipeline to an audio tensor."""
        if torch.rand(1) < self.p:
            if len(audio_tensor.shape) == 1:
                samples_3d = audio_tensor.unsqueeze(0).unsqueeze(0)
            elif len(audio_tensor.shape) == 2:
                samples_3d = audio_tensor.unsqueeze(0)
            else:
                samples_3d = audio_tensor

            if samples_3d.shape[1] > 1:
                samples_3d = samples_3d[:, 0, :].unsqueeze(1)

            processed_audio = self.augmentations(samples=samples_3d, sample_rate=self.sample_rate)

            return processed_audio['samples'].squeeze()

        return audio_tensor