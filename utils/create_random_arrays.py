#!/usr/bin/env python3
import argparse
import numpy as np

def parse_shapes(shape_args):
    """
    2x3 4x5 1x6 => [(2,3), (4,5), (1,6)]
    """
    shapes = []
    for s in shape_args:
        dims = tuple(map(int, s.lower().split("x")))
        shapes.append(dims)
    return shapes


def create_random_arrays(shapes, integer=False, low=0, high=10):
    """
    Create random NumPy arrays with different shapes
    """
    arrays = []
    for shape in shapes:
        if integer:
            arr = np.random.randint(low, high, size=shape)
        else:
            arr = np.random.rand(*shape)
        arrays.append(arr)
    return arrays


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random NumPy arrays with possibly different shapes.")
    parser.add_argument("shapes", nargs="+", 
                        help="List of shapes, e.g. 2x3 4x5 6x1")
    parser.add_argument("--integer", action="store_true",
                        help="Generate integers instead of floats.")
    parser.add_argument("--low", type=int, default=0,
                        help="Lower bound for integers (inclusive).")
    parser.add_argument("--high", type=int, default=10,
                        help="Upper bound for integers (exclusive).")

    np.random.seed(42)

    args = parser.parse_args()

    shapes = parse_shapes(args.shapes)

    arrays = create_random_arrays(
        shapes=shapes,
        integer=args.integer,
        low=args.low,
        high=args.high
    )

    for i, arr in enumerate(arrays, 1):
        print(f"Array {i} with shape {arr.shape}:\n{arr}\n")
