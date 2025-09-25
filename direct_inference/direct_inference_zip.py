import h5py
import numpy as np
from stable_baselines3 import PPO  # or A2C, SAC, etc. — depends on your model

# Path to your saved model
model_path = "model_final.zip"

# Load SB3 model
model = PPO.load(model_path)  # replace PPO with the correct algo

# Load your inputs
with h5py.File("inputs.h5", "r") as hf:
    Input_img = np.array(hf["img"], dtype=np.float32)   # shape (4, 4, 36, 64)
    Input_vec = np.array(hf["vec"], dtype=np.float32)   # shape (4, 12)

# SB3 expects a single observation that matches its training space
# Since your model was trained with a dict observation {"img":..., "vec":...},
# you need to pass that format.
obs = {
    "img": Input_img[0],   # one sample
    "vec": Input_vec[0]
}

# Run inference
action, _states = model.predict(obs, deterministic=True)

print("Predicted action:", action)
