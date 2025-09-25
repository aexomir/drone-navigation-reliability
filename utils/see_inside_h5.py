import h5py
import argparse
import numpy as np

def inspect_h5(file_path):
    with h5py.File(file_path, "r") as f:
        print(f"Keys in '{file_path}': {list(f.keys())}\n")

        for key in f.keys():
            data = f[key][:]
            print(f"Dataset '{key}':")
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            print(f"  Data:\n{data}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect contents of an HDF5 file.")
    parser.add_argument("h5file", type=str, help="Path to the .h5 file to inspect")

    args = parser.parse_args()
    inspect_h5(args.h5file)
