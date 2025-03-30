import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuration ---
SIMULATIONS_DIR = "../data/simulations/"
SAMPLE_RATE = 44100

def verify_simulation(sim_dir):
    """Verify one simulation directory (e.g., gunshot_0/)"""
    print(f"\nVerifying {sim_dir}...")
    
    # Load all 6 mic recordings
    recordings = []
    durations = []
    for mic in range(1, 7):
        file = os.path.join(sim_dir, f"mic_{mic}_recording.wav")
        audio, sr = librosa.load(file, sr=SAMPLE_RATE)
        recordings.append(audio)
        durations.append(len(audio)/sr)
    
    # Check 1: All mics have same duration
    assert len(set(durations)) == 1, f"Inconsistent durations: {durations}"
    print(f"✓ All mics have same duration: {durations[0]:.2f}s")
    
    # Check 2: Verify delays (mic1 should be earliest)
    max_length = max(len(r) for r in recordings)
    aligned = [np.pad(r, (0, max_length - len(r))) for r in recordings]
    cross_corr = np.array([[np.corrcoef(aligned[i], aligned[j])[0,1] 
                          for j in range(6)] for i in range(6)])
    
    plt.figure(figsize=(10, 4))
    plt.imshow(cross_corr, vmin=0.5, vmax=1, cmap='viridis')
    plt.colorbar(label='Correlation')
    plt.title("Cross-Correlation Between Mics")
    plt.xlabel("Mic Index"); plt.ylabel("Mic Index")
    plt.show()
    
    # Check 3: Plot first 1000 samples (should show delays)
    plt.figure(figsize=(12, 6))
    for i, audio in enumerate(recordings):
        plt.plot(audio[:1000] + i*0.5, label=f"Mic {i+1}")
    plt.title("First 1000 Samples (Offset for Clarity)")
    plt.xlabel("Samples"); plt.ylabel("Amplitude (Offset)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Verify all simulations
    sim_folders = sorted([os.path.join(SIMULATIONS_DIR, d) 
                         for d in os.listdir(SIMULATIONS_DIR) 
                         if d.startswith("gunshot_")])
    
    for sim_dir in tqdm(sim_folders[:3]):  # Check first 3 simulations
        verify_simulation(sim_dir)