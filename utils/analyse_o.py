import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse

def visualize_analysis(pred_file, exp_file, pred_key, exp_key, out_file="analysis_results.png"):
    # Load datasets
    with h5py.File(pred_file, "r") as f_pred, h5py.File(exp_file, "r") as f_exp:
        if pred_key not in f_pred:
            raise KeyError(f"Key '{pred_key}' not found in predicted file")
        if exp_key not in f_exp:
            raise KeyError(f"Key '{exp_key}' not found in expected file")

        pred = f_pred[pred_key][:]
        exp = f_exp[exp_key][:]

    # Shape check
    if pred.shape != exp.shape:
        raise ValueError(f"Shape mismatch: predicted {pred.shape}, expected {exp.shape}")

    # Convert dtype if needed
    # if pred.dtype != exp.dtype:
    #     pred = pred.astype(np.float32)
    #     exp = exp.astype(np.float32)

    # Flatten for plotting
    pred_flat = pred.flatten()
    exp_flat = exp.flatten()
    print(f'pred: {pred_flat} expected: {exp_flat}')
    errors = pred_flat - exp_flat

    # Metrics
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}")

    # Plots
    plt.figure(figsize=(12, 5))

    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(exp_flat, pred_flat, alpha=0.5, s=10)
    min_val, max_val = min(exp_flat.min(), pred_flat.min()), max(exp_flat.max(), pred_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")  # ideal diagonal
    plt.xlabel("Expected")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Expected")

    # Error histogram
    plt.subplot(1, 2, 2)
    plt.hist(errors, bins=30, color="skyblue", edgecolor="black")
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Prediction Error (Pred - Exp)")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")

    plt.tight_layout()
    plt.savefig(out_file)
    print(f"Visualization saved to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize predicted vs expected outputs from H5 files.")
    parser.add_argument("pred_file", help="Path to predicted outputs H5 file")
    parser.add_argument("exp_file", help="Path to expected outputs H5 file")
    parser.add_argument("--pred_key", required=True, help="Dataset key inside the predicted H5 file")
    parser.add_argument("--exp_key", required=True, help="Dataset key inside the expected H5 file")
    parser.add_argument("--out", default="analysis_results.png", help="Output PNG file name")
    args = parser.parse_args()

    visualize_analysis(args.pred_file, args.exp_file, args.pred_key, args.exp_key, args.out)