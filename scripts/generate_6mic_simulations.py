import os
import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import librosa
from tqdm import tqdm
import pandas as pd

# --- Configuration ---
SAMPLE_RATE = 44100
MIC_DISTANCE = 1.5  # meters
ROOM_DIM = np.array([10, 10, 3])  # 10m x 10m x 3m room
NUM_SIMULATIONS = 100
NOISE_SNR = 15  # dB

# --- Paths ---
SYNTHETIC_GUNSHOTS_DIR = "../data/synthetic_gunshots/"
OUTPUT_DIR = "../data/simulations/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Hexagonal Microphone Array (centered in room) ---
mic_positions = np.array([
    [0, 0, 0],               # Mic 1 (center)
    [MIC_DISTANCE, 0, 0],     # Mic 2 (right)
    [MIC_DISTANCE/2, MIC_DISTANCE*np.sqrt(3)/2, 0],  # Mic 3 (top-right)
    [-MIC_DISTANCE/2, MIC_DISTANCE*np.sqrt(3)/2, 0], # Mic 4 (top-left)
    [-MIC_DISTANCE, 0, 0],    # Mic 5 (left)
    [-MIC_DISTANCE/2, -MIC_DISTANCE*np.sqrt(3)/2, 0] # Mic 6 (bottom-left)
]).T  # Shape: (3, 6)

# Offset mics to room center
mic_positions += ROOM_DIM[:, None] / 2

# --- Load Gunshots ---
gunshot_files = [f for f in os.listdir(SYNTHETIC_GUNSHOTS_DIR) if f.endswith(".wav")]
gunshots = []
for file in gunshot_files:
    audio, _ = librosa.load(os.path.join(SYNTHETIC_GUNSHOTS_DIR, file), sr=SAMPLE_RATE)
    gunshots.append(audio)

# --- Simulation Function ---
def simulate_gunshot(source_pos, gunshot_audio):
    """Simulate gunshot from source_pos inside the room."""
    room = pra.ShoeBox(ROOM_DIM, fs=SAMPLE_RATE, materials=pra.Material(0.2), max_order=10)
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, SAMPLE_RATE))
    
    # Ensure source is inside room boundaries
    if not np.all((0 <= source_pos) & (source_pos <= ROOM_DIM)):
        raise ValueError(f"Source position {source_pos} is outside room dimensions {ROOM_DIM}")
    
    room.add_source(source_pos, signal=gunshot_audio)
    room.simulate()
    return room.mic_array.signals

# --- Generate Data ---
labels = []
for sim_idx in tqdm(range(NUM_SIMULATIONS), desc="Simulating gunshots"):
    # Randomize gunshot position INSIDE ROOM (min 1m from walls)
    margin = 1.0
    valid_pos = False
    while not valid_pos:
        distance = np.random.uniform(5, min(ROOM_DIM)/2 - margin)
        azimuth = np.random.uniform(-180, 180)
        elevation = np.random.uniform(-30, 30)
        
        # Convert to Cartesian (relative to Mic1 at room center)
        azimuth_rad = np.radians(azimuth)
        elevation_rad = np.radians(elevation)
        x = distance * np.cos(azimuth_rad) * np.cos(elevation_rad)
        y = distance * np.sin(azimuth_rad) * np.cos(elevation_rad)
        z = distance * np.sin(elevation_rad)
        source_pos = np.array([x, y, z]) + ROOM_DIM / 2
        
        # Check if inside room
        valid_pos = np.all((margin <= source_pos) & (source_pos <= ROOM_DIM - margin))
    
    # Simulate
    gunshot_audio = gunshots[np.random.randint(0, len(gunshots))]
    recordings = simulate_gunshot(source_pos, gunshot_audio)
    
    # Add noise
    noise = np.random.randn(*recordings.shape)
    signal_power = np.mean(recordings ** 2)
    noise_scale = np.sqrt(signal_power / (10 ** (NOISE_SNR/10)))
    recordings += noise * noise_scale
    
    # Save recordings
    sim_dir = os.path.join(OUTPUT_DIR, f"gunshot_{sim_idx}")
    os.makedirs(sim_dir, exist_ok=True)
    for mic_idx in range(6):
        sf.write(
            os.path.join(sim_dir, f"mic_{mic_idx+1}_recording.wav"),
            recordings[mic_idx],
            SAMPLE_RATE
        )
    
    # Store labels
    labels.append({
        "simulation_id": sim_idx,
        "distance": distance,
        "azimuth": azimuth,
        "elevation": elevation,
        "source_x": x,
        "source_y": y,
        "source_z": z
    })

# Save labels
pd.DataFrame(labels).to_csv(os.path.join(OUTPUT_DIR, "labels.csv"), index=False)
print(f"Generated {NUM_SIMULATIONS} simulations in {OUTPUT_DIR}")