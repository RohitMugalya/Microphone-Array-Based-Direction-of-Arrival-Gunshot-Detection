import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# --- Configuration ---
SAMPLE_RATE = 44100
SNR_RANGE = (5, 20)  # Min and max SNR (dB)
OUTPUT_DIR = "../data/synthetic_gunshots/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load Original Data (Variable Length) ---
def load_audio_files(folder_path):
    """Load all .wav files without enforcing fixed duration."""
    audio_files = []
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
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
    
    # Trim noise to match gunshot length
    min_length = min(len(gunshot), len(noise))
    gunshot = gunshot[:min_length]
    noise = noise[:min_length]
    
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

# --- Verification ---
print("\nExample synthetic audio shapes:")
for i in range(min(3, len(gunshots))):
    print(f"Original gunshot {i}: {len(gunshots[i])} samples")
    print(f"Synthetic gunshot {i}: {len(synthetic_audio)} samples")