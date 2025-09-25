import os
import numpy as np
import h5py
from stable_baselines3 import DQN

# --- Configuration ---
MODEL_PATH = "../nvbitfi/model_final.zip"
H5_INPUT_PATH = "inputs.h5"

# --- 1. Load the model ---
try:
    # Use the correct SB3 algorithm class here (PPO, A2C, SAC, etc.)
    model = DQN.load(MODEL_PATH)
    print(f"✅ Successfully loaded SB3 model from {MODEL_PATH}.")
except Exception as e:
    print(f"❌ Error loading model. Ensure {MODEL_PATH} exists and is a valid SB3 zip file, and the correct algorithm class is used.")
    print(f"Error details: {e}")
    exit()

with h5py.File(H5_INPUT_PATH, 'r') as hf:
    img_data = hf['img'][:].astype(np.float32)
    vec_data = hf['vec'][:].astype(np.float32)

print(f"  - img shape: {img_data.shape}, dtype: {img_data.dtype}")
print(f"  - vec shape: {vec_data.shape}, dtype: {vec_data.dtype}")

observation_batch = {
    "img": img_data,
    "vec": vec_data,
}

actions, _ = model.predict(observation_batch, deterministic=True)

print(actions)