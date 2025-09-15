import argparse
import h5py

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect contents of an HDF5 file.")
    parser.add_argument("file", type=str, help="Path to the HDF5 file")
    args = parser.parse_args()

    with h5py.File(args.file, "r") as f:
        print(f"Contents of {args.file}:")
        for name, dataset in f.items():
            print(f" - {name}: shape={dataset.shape}, dtype={dataset.dtype}")