import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "alexnet"
RSA_DIR = PROJECT_ROOT / "outputs/clean_baseline/rsa" / MODEL_NAME
SAVE_PATH = PROJECT_ROOT / "outputs/clean_baseline/alexnet_final_dashboard.png"
SAVE_PATH_BEHAVIOR = PROJECT_ROOT / "outputs/clean_baseline/alexnet_behavioral_comparison.png"

LAYERS = ["features.2", "features.5", "features.7", "features.9", "features.12", 
          "classifier.2", "classifier.5", "classifier.6"]

def smooth_data(data, window=10):
    """Smooths jittery MEG data to find the true peak."""
    smoothing = np.convolve(data, np.ones(window)/window, mode='same')
    return smoothing

def main():
    # Set up a two-panel figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    colors = plt.cm.viridis(np.linspace(0, 1, len(LAYERS)))
    
    # Set up a figure for behavioral model comparisons
    fig2, axes2 = plt.subplots(3, 1, figsize=(8, 10))

    # Store data for the bottom plot
    final_peaks = []
    final_errors = []
    
    # Setup time mapping
    search_start_ms = 50
    search_end_ms = 450
    n_bootstraps = 1000 # 1000 is plenty for stable error bars without taking hours to run
    
    # 1. Plot the Time Courses (Top Panel)
    for i, layer in enumerate(LAYERS):
        safe_layer = layer.replace(".", "_").replace("/", "_")
        mean_file = RSA_DIR / f"{safe_layer}_rsa_spearman_mean.npy"
        subj_file = RSA_DIR / f"{safe_layer}_rsa_spearman_subjects.npy"
        
        if mean_file.exists() and subj_file.exists():
            # Load data
            mean_data = np.load(mean_file)
            subj_data = np.load(subj_file) # Shape: (Time, Subjects)
            n_time, n_subj = subj_data.shape
            
            time_points = np.linspace(-200, 1000, n_time)
            start_idx = np.argmin(np.abs(time_points - search_start_ms))
            end_idx = np.argmin(np.abs(time_points - search_end_ms))
            
            # --- PAPER METHOD: BOOTSTRAPPING ---
            bootstrapped_peaks = []
            
            for _ in range(n_bootstraps):
                # Randomly sample subjects with replacement
                sample_indices = np.random.choice(n_subj, size=n_subj, replace=True)
                boot_sample = subj_data[:, sample_indices]
                
                # Average and smooth this specific bootstrap sample
                boot_mean = boot_sample.mean(axis=1)
                boot_smoothed = smooth_data(boot_mean)
                
                # Find the peak for this sample
                window_segment = boot_smoothed[start_idx:end_idx]
                peak_idx = start_idx + np.argmax(window_segment)
                bootstrapped_peaks.append(time_points[peak_idx])
            
            # The final peak is the mean of all bootstrap peaks, and error is the standard dev
            layer_peak_mean = np.mean(bootstrapped_peaks)
            layer_peak_err = np.std(bootstrapped_peaks)
            
            final_peaks.append(layer_peak_mean)
            final_errors.append(layer_peak_err)
            # -----------------------------------

            # Plot top panel (using standard mean data)
            smoothed_mean = smooth_data(mean_data)
            ax1.plot(time_points, smoothed_mean, label=f"{layer}", color=colors[i], linewidth=2)
        else:
            print(f"Missing files for: {layer}. Make sure you ran the subject-level RSA script!")
            final_peaks.append(np.nan)
            final_errors.append(np.nan)

    ax1.set_title(f"Brain-AI Correlation Time-Course ({MODEL_NAME.upper()})", fontsize=14)
    ax1.set_ylabel("Spearman Correlation")
    ax1.set_xlim([-200, 1000])
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label="Stimulus Onset (0ms)")
    ax1.legend(loc='upper right', fontsize='small', ncol=2)
    ax1.grid(alpha=0.2)

    # 2. Plot the Latencies (Bottom Panel)
    layer_indices = range(1, len(LAYERS) + 1)
    ax2.plot(layer_indices, final_peaks, '-', color='black', linewidth=1.5, zorder=1)
    
    # Plot the dots with standard error bars from the bootstrapping
    for i in range(len(LAYERS)):
        if not np.isnan(final_peaks[i]):
            ax2.errorbar(i+1, final_peaks[i], yerr=final_errors[i], fmt='o', 
                         color=colors[i], markersize=10, capsize=5, capthick=2, linewidth=2, zorder=2)

    ax2.set_title("Peak Latency per Layer (Bootstrapped SEM)", fontsize=14)
    ax2.set_xticks(layer_indices)
    ax2.set_xticklabels(LAYERS, rotation=45)
    ax2.set_ylabel("Time of Peak Match (ms)")
    ax2.set_xlabel("AlexNet Layers")
    ax2.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"Dashboard saved to: {SAVE_PATH}")

    # 3. Plot Behavioral Model Comparisons
    behavioral_models = ["semantic", "structure", "visual"]
    for i, model in enumerate(behavioral_models):
        model_file = RSA_DIR / f"{model}_rsa_spearman.npy"
        if model_file.exists():
            model_data = np.load(model_file).flatten()
            print(model_data)
            axes2[i].bar(np.arange(len(model_data)), model_data, color=colors[:len(model_data)])
            axes2[i].set_title(f"RSA with {model.capitalize()} Model", fontsize=12)
            axes2[i].set_xticks(np.arange(len(LAYERS)))
            axes2[i].set_xticklabels(LAYERS, rotation=25)
            axes2[i].set_ylabel("Spearman Correlation")
        else:
            print(f"Missing behavioral model data for: {model}")

    plt.tight_layout()
    plt.savefig(SAVE_PATH_BEHAVIOR)
    print(f"Dashboard saved to: {SAVE_PATH_BEHAVIOR}")
    plt.show()

if __name__ == "__main__":
    main()