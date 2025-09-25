import os
import argparse
import traceback
from collections import OrderedDict

import torch
import torch.nn as nn
import onnx
import tensorrt as trt
import stable_baselines3 as sb3
import gymnasium as gym
import numpy as np

# --- Configuration ---
MODEL_ZIP_PATH        = "./model_final.zip"
ONNX_MODEL_PATH       = "model_final.onnx"
TENSORRT_ENGINE_PATH  = "model_final.trt"
SB3_ALGORITHM_CLASS   = sb3.DQN
TRT_LOGGER            = trt.Logger(trt.Logger.WARNING)
MAX_WORKSPACE_SIZE    = 1 << 30
USE_FP16              = False

# --- Helper Functions ---
def check_onnx_model(onnx_path):
    """Checks the validity of the ONNX model."""
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"ONNX model {onnx_path} checked successfully.")
        return True
    except Exception as e:
        print(f"Error checking ONNX model {onnx_path}: {e}")
        print(traceback.format_exc())
        return False

def build_tensorrt_engine(onnx_path, engine_path, logger, workspace_size, use_fp16):
    """Builds the TensorRT engine from an ONNX file with dynamic-shape optimization profiles."""
    if os.path.exists(engine_path):
        print(f"TensorRT engine {engine_path} already exists. Skipping build.")
        return True

    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    config  = builder.create_builder_config()
    parser  = trt.OnnxParser(network, logger)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

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

    # ─── Define an optimization profile for dynamic batch sizes ───
    profile = builder.create_optimization_profile()
    # Here: “img” shape is [batch, 4, 36, 64], “vec” is [batch, 12]
    profile.set_shape("img",
                      min=(1, 4, 36, 64),
                      opt=(4, 4, 36, 64),
                      max=(8, 4, 36, 64))
    profile.set_shape("vec",
                      min=(1, 12),
                      opt=(4, 12),
                      max=(8, 12))
    config.add_optimization_profile(profile)
    # ────────────────────────────────────────────────────────────────

    print(f"Building TensorRT engine (this may take a while)...")
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        print("ERROR: Failed to build the TensorRT engine.")
        return False

    print("TensorRT engine built successfully.")
    with open(engine_path, "wb") as f:
        f.write(plan)
    print(f"TensorRT engine saved to: {engine_path}")
    return True

# --- Wrapper Module ---
class PolicyWrapper(nn.Module):
    """
    Wraps the SB3 policy to accept separate tensors for 'img' and 'vec'
    so we can export it cleanly to ONNX.
    """
    def __init__(self, policy):
        super().__init__()
        self.policy = policy.eval()

    def forward(self, img, vec):
        obs_dict = OrderedDict([('img', img), ('vec', vec)])
        # For DQN, calling the policy returns Q-values
        # AEXO-prev: return self.policy(obs_dict)
        result = torch.argmax(self.policy(obs_dict), dim=1)
        return result

# --- Main Conversion Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SB3 model to ONNX + TensorRT (dynamic shapes).")
    parser.add_argument("--skip_onnx", action="store_true", help="Skip ONNX export if already done.")
    parser.add_argument("--skip_trt",  action="store_true", help="Skip TensorRT engine build.")
    parser.add_argument("--bs", help="Add Batch Size.", default=1)
    args = parser.parse_args()

    # Step 1: Export to ONNX
    if not args.skip_onnx or not os.path.exists(ONNX_MODEL_PATH):
        print("--- Step 1: Exporting to ONNX ---")
        if not os.path.exists(MODEL_ZIP_PATH):
            print(f"ERROR: Model zip not found at {MODEL_ZIP_PATH}")
            exit(1)

        try:
            print(f"Loading SB3 model from {MODEL_ZIP_PATH}...")
            model = SB3_ALGORITHM_CLASS.load(MODEL_ZIP_PATH, device="cpu")
            print("Model loaded successfully.")

            # Wrap policy for ONNX export
            # AEXO-PREV: PolicyWrapper(model.policy)
            policy_wrapper = PolicyWrapper(model.policy.q_net)
            img =torch.tensor(np.random.rand(32,4,36,64))
            vec = torch.tensor(np.random.rand(32,12,1))
            output = policy_wrapper(img, vec)
            # print(output.shape)
            policy_wrapper.eval()
            print("Using PolicyWrapper for ONNX export.")

            # Build dummy inputs
            obs_space = model.observation_space
            if isinstance(obs_space, gym.spaces.Dict):
                # must match wrapper args: img, vec
                names = ["img", "vec"]
                tensors = []
                for key in names:
                    space = obs_space.spaces[key]
                    shape = (int(args.bs),) + space.shape
                    tensors.append(torch.randn(shape, dtype=torch.float32))
                    print(f"Dummy '{key}' tensor: {shape}")
                dummy_inputs = tuple(tensors)
            else:
                print(f"ERROR: Unsupported space {obs_space}")
                exit(1)

            # Export
            print(f"Exporting to ONNX at {ONNX_MODEL_PATH}...")
            torch.onnx.export(
                policy_wrapper,
                dummy_inputs,
                ONNX_MODEL_PATH,
                # export_params=True,
                # opset_version=11,
                # do_constant_folding=True,
                # input_names=names,
                # output_names=["output_actions"],
                # dynamic_axes={
                #     "img": {0: "batch_size"},
                #     "vec": {0: "batch_size"},
                #     "output_actions": {0: "batch_size"}
                # }
            )
            print("ONNX export complete.")
            check_onnx_model(ONNX_MODEL_PATH)

        except Exception as e:
            print(f"ERROR during ONNX export: {e}")
            print(traceback.format_exc())
            exit(1)
    else:
        print("--- Skipping ONNX export (exists or skipped) ---")
        if not os.path.exists(ONNX_MODEL_PATH):
            print(f"ERROR: {ONNX_MODEL_PATH} not found.")
            exit(1)
        check_onnx_model(ONNX_MODEL_PATH)

    # Step 2: Build TensorRT engine
    if not args.skip_trt:
        print("\n--- Step 2: Building TensorRT Engine ---")
        if not build_tensorrt_engine(ONNX_MODEL_PATH, TENSORRT_ENGINE_PATH,
                                     TRT_LOGGER, MAX_WORKSPACE_SIZE, USE_FP16):
            print("TensorRT engine build failed.")
            exit(1)
        print("TensorRT engine build succeeded.")
    else:
        print("--- Skipping TensorRT build (skipped via flag) ---")

    print("\nConversion script finished.")
