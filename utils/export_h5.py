#!/usr/bin/env python3
import argparse
import h5py
import os
from create_random_arrays import create_random_arrays

def parse_named_shapes(named_shape_args):
    """
    Parse arguments like: name1:2x3 name2:4x5x6
    into [("name1", (2,3)), ("name2", (4,5,6))]
    """
    parsed = []
    for arg in named_shape_args:
        if ":" not in arg:
            raise ValueError(f"Invalid format '{arg}'. Expected name:shape (e.g. myarr:2x3)")
        name, shape_str = arg.split(":", 1)
        shape = tuple(map(int, shape_str.lower().split("x")))
        parsed.append((name, shape))
    return parsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random arrays and export to HDF5 with custom names.")
    parser.add_argument("output", type=str, help="Output HDF5 file path (e.g. arrays.h5)")
    parser.add_argument("named_shapes", nargs="+",
                        help="List of name:shape pairs, e.g. arr1:2x3 arr2:4x5x6")
    parser.add_argument("--integer", action="store_true",
                        help="Generate integers instead of floats.")
    parser.add_argument("--low", type=int, default=0,
                        help="Lower bound for integers (inclusive).")
    parser.add_argument("--high", type=int, default=10,
                        help="Upper bound for integers (exclusive).")

    args = parser.parse_args()

    # Parse names and shapes
    named_shapes = parse_named_shapes(args.named_shapes)

    # Just pass the shapes to the generator
    shapes = [shape for _, shape in named_shapes]
    arrays = create_random_arrays(
        shapes=shapes,
        integer=args.integer,
        low=args.low,
        high=args.high
    )

    # Save with custom names
    with h5py.File(args.output, "w") as f:
        for (name, _), arr in zip(named_shapes, arrays):
            f.create_dataset(name, data=arr)

    print(f"Saved {len(arrays)} arrays to {os.path.abspath(args.output)}")
    for (name, shape) in named_shapes:
        print(f" - {name}: shape={shape}")
