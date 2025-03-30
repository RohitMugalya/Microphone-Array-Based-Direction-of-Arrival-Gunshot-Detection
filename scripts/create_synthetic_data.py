import os
import numpy as np
import librosa
from tqdm import tqdm
import soundfile as sf

# --- Configuration ---
SAMPLE_RATE = 44100
DURATION = 1.0  # Seconds
SNR_RANGE = (5, 20)  # Min and max SNR (dB)
OUTPUT_DIR = "../data/synthetic_gunshots/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load Original Data ---
def load_audio_files(folder_path):
    """Load all .wav files from a folder and trim/pad to fixed duration."""
    audio_files = []
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
            if len(audio) < int(SAMPLE_RATE * DURATION):
                audio = np.pad(audio, (0, int(SAMPLE_RATE * DURATION) - len(audio)))
            else:
                audio = audio[:int(SAMPLE_RATE * DURATION)]
            audio_files.append(audio)
    return audio_files

gunshots = load_audio_files("../data/gunshots/")
noises = load_audio_files("../data/non_gunshots/")

# --- Synthetic Data Generation ---
def mix_gunshot_noise(gunshot, noise, snr_db):
    """Mix gunshot with noise at specified SNR."""
    # Normalize both signals
    gunshot = gunshot / np.max(np.abs(gunshot))
    noise = noise / np.max(np.abs(noise))
    
    # Scale noise to achieve desired SNR
    gunshot_power = np.mean(gunshot ** 2)
    noise_power = np.mean(noise ** 2)
    scale = np.sqrt(gunshot_power / (noise_power * (10 ** (snr_db / 10))))
    noise_scaled = noise * scale
    
    # Mix and renormalize
    mixed = gunshot + noise_scaled
    return mixed / np.max(np.abs(mixed))

# Generate synthetic data
for i, gunshot in enumerate(tqdm(gunshots, desc="Generating synthetic data")):
    # Randomly select noise and SNR
    noise = noises[np.random.randint(0, len(noises))]
    snr_db = np.random.uniform(SNR_RANGE[0], SNR_RANGE[1])
    
    # Mix gunshot + noise
    synthetic_audio = mix_gunshot_noise(gunshot, noise, snr_db)
    
    # Save to synthetic_gunshots/
    output_path = os.path.join(OUTPUT_DIR, f"synthetic_gunshot_{i}.wav")
    sf.write(output_path, synthetic_audio, SAMPLE_RATE)

print(f"Synthetic data saved to {OUTPUT_DIR}")