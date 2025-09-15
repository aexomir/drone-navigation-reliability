import os
import argparse

import tensorrt as trt


# --- Defaults (kept consistent with convert.py) ---
DEFAULT_WORKSPACE_SIZE = 1 << 30
DEFAULT_USE_FP16 = False


def build_tensorrt_engine(onnx_path, engine_path, logger, workspace_size, use_fp16):
    """Builds the TensorRT engine from an ONNX file with dynamic-shape optimization profiles.

    The logic mirrors the implementation in convert.py to ensure identical behavior.
    """
    if os.path.exists(engine_path):
        print(f"TensorRT engine {engine_path} already exists. Skipping build.")
        return True

    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, logger)

    # Workspace memory (TRT >= 8.5 uses set_memory_pool_limit; TRT 8.2 uses max_workspace_size)
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    else:
        config.max_workspace_size = int(workspace_size)

    if use_fp16:
        if builder.platform_has_fast_fp16:
            print("Enabling FP16 precision.")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("FP16 not supported on this platform, using FP32.")

    if not os.path.exists(onnx_path):
        print(f"ONNX file not found: {onnx_path}")
        return False

    print(f"Loading ONNX file from path {onnx_path}...")
    with open(onnx_path, "rb") as model_file:
        print("Parsing ONNX model...")
        success = parser.parse(model_file.read())
        if not success:
            print("ERROR: Failed to parse the ONNX file.")
            for idx in range(parser.num_errors):
                print(f"Parser Error {idx}: {parser.get_error(idx)}")
            return False
        print("ONNX model parsed successfully.")

    # Show inputs/outputs
    print(f"Network inputs: {[network.get_input(i).name for i in range(network.num_inputs)]}")
    print(f"Network outputs: {[network.get_output(i).name for i in range(network.num_outputs)]}")

    # Define an optimization profile for dynamic batch sizes
    profile = builder.create_optimization_profile()
    # Here: “img” shape is [batch, 4, 36, 64], “vec” is [batch, 12]
    profile.set_shape(
        "img",
        min=(1, 4, 36, 64),
        opt=(4, 4, 36, 64),
        max=(8, 4, 36, 64),
    )
    profile.set_shape(
        "vec",
        min=(1, 12),
        opt=(4, 12),
        max=(8, 12),
    )
    config.add_optimization_profile(profile)

    print("Building TensorRT engine (this may take a while)...")
    plan = None
    if hasattr(builder, "build_serialized_network"):
        # TRT >= 8.4
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            print("ERROR: Failed to build the TensorRT engine.")
            return False
        with open(engine_path, "wb") as f:
            # plan is already serialized bytes-like
            try:
                f.write(plan)
            except TypeError:
                f.write(bytes(plan))
    else:
        # TRT 8.2 fallback: build ICudaEngine then serialize
        engine = builder.build_engine(network, config)
        if engine is None:
            print("ERROR: Failed to build the TensorRT engine.")
            return False
        serialized = engine.serialize()
        with open(engine_path, "wb") as f:
            try:
                f.write(serialized)
            except TypeError:
                # IHostMemory -> use buffer/bytes
                if hasattr(serialized, "buffer"):
                    f.write(serialized.buffer)
                else:
                    f.write(bytes(serialized))
    print("TensorRT engine built successfully.")
    print(f"TensorRT engine saved to: {engine_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build a TensorRT engine from an ONNX model using the existing logic."
    )
    parser.add_argument(
        "--onnx",
        required=True,
        help="Path to the input ONNX model",
    )
    parser.add_argument(
        "--engine",
        required=False,
        default="model_final.trt",
        help="Output path for the TensorRT engine (.trt)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=DEFAULT_USE_FP16,
        help="Enable FP16 precision if supported",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=DEFAULT_WORKSPACE_SIZE,
        help="Workspace size in bytes (default: 1<<30)",
    )
    args = parser.parse_args()

    trt_logger = trt.Logger(trt.Logger.WARNING)

    print("\n--- Building TensorRT Engine from ONNX ---")
    success = build_tensorrt_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        logger=trt_logger,
        workspace_size=args.workspace,
        use_fp16=args.fp16,
    )
    if not success:
        raise SystemExit(1)
    print("TensorRT engine build succeeded.")


if __name__ == "__main__":
    main()


