import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "alexnet"
RSA_DIR = PROJECT_ROOT / "outputs/clean_baseline/rsa" / MODEL_NAME
SAVE_PATH = PROJECT_ROOT / "outputs/clean_baseline/alexnet_final_dashboard.png"

LAYERS = ["features.2", "features.5", "features.7", "features.9", "features.12", 
          "classifier.2", "classifier.5", "classifier.6"]

def smooth_data(data, window=30):
    """Smooths jittery MEG data to find the true peak."""
    return np.convolve(data, np.ones(window)/window, mode='same')

def main():
    # Set up a two-panel figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    colors = plt.cm.viridis(np.linspace(0, 1, len(LAYERS)))
    
    peak_latencies = []
    
    # 1. Plot the Time Courses (Top Panel)
    for i, layer in enumerate(LAYERS):
        safe_layer = layer.replace(".", "_").replace("/", "_")
        file_path = RSA_DIR / f"{safe_layer}_rsa_spearman.npy"
        
        if file_path.exists():
            data = np.load(file_path)
            smoothed = smooth_data(data)
            time_points = np.arange(len(data))
            
            # --- THE FIX: SEARCH WINDOW ---
            # Define window to ignore pre-stimulus and end-of-trial noise
            # index 50 = 50ms, index 600 = 600ms
            start, end = 50, 1000 
            
            # Find the peak ONLY within this biologically relevant window
            window_segment = smoothed[start:end]
            peak_in_window = np.argmax(window_segment)
            actual_peak_milli = peak_in_window + start
            
            peak_latencies.append(actual_peak_milli)
            # ------------------------------

            ax1.plot(time_points, smoothed, label=f"{layer} (Peak: {actual_peak_milli}ms)", color=colors[i], linewidth=2)
        else:
            print(f"Missing: {file_path}")
            peak_latencies.append(np.nan)

    ax1.set_title(f"Brain-AI Correlation Time-Course ({MODEL_NAME.upper()})", fontsize=14)
    ax1.set_ylabel("Spearman Correlation")
    ax1.axvline(x=230, color='red', linestyle='--', alpha=0.5, label="230ms Event")
    ax1.legend(loc='upper right', fontsize='small', ncol=2)
    ax1.grid(alpha=0.2)

    # 2. Plot the Latencies (Bottom Panel)
    layer_indices = range(1, len(LAYERS) + 1)
    ax2.plot(layer_indices, peak_latencies, 'o-', color='black', linewidth=2, markersize=8)
    
    # Color the dots to match the lines above
    for i, lat in enumerate(peak_latencies):
        ax2.plot(i+1, lat, 'o', color=colors[i], markersize=10)

    ax2.set_title("Peak Latency per Layer (Early -> Late)", fontsize=14)
    ax2.set_xticks(layer_indices)
    ax2.set_xticklabels(LAYERS, rotation=45)
    ax2.set_ylabel("Time of Peak Match (ms)")
    ax2.set_xlabel("AlexNet Layers")
    ax2.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"Dashboard saved to: {SAVE_PATH}")
    plt.show()

if __name__ == "__main__":
    main()