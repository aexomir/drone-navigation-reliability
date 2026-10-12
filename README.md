# RL Drone Navigation: Fault-Injection Reliability Testing on Jetson Nano

Research project at Politecnico di Torino, supervised by Prof. Matteo Sonza Reorda. It looks at how hardware faults affect a reinforcement-learning policy for drone navigation when the policy runs on embedded hardware.

**Status: unfinished.** The model-conversion pipeline and the layer-level test harness exist. The fault-injection campaign and its results do not. There are no reliability numbers in this repository.

## What is here

| File | Purpose |
|---|---|
| `conv_from_zip_to_onnx.py` | Loads the trained DQN policy (Stable-Baselines3 `.zip`) and exports it to ONNX. By default it then also builds a TensorRT engine from it (skip with `--skip_trt`). |
| `conv_from_onnx_to_trt.py` | Builds a TensorRT engine from an ONNX model passed with `--onnx` (output: `model_final.trt` by default). |
| `Run_Layer.py` | Command-line entry point. Runs one layer of the network, either from a TensorRT engine (`-trt`, via `nvbitfi_DNN_TRT.py`) or through the `nvbitfi_DNN` module, which with `-onnx` also exports the layer to ONNX. |
| `run.sh` | Wrapper that runs `Run_Layer.py` and renames its output to `Golden_Output_layer.h5`, so later runs can be compared against it. The rename always happens, because `GOLDEN_FLAG` is hardcoded to `1`; use it for the fault-free run. |
| `utils/` | Helpers: create random input arrays, write and inspect HDF5 files, measure RAM, GPU load and temperature on the Jetson across batch sizes. |
| `model_final.zip`, `.onnx`, `.trt` | The trained policy and its converted forms. |
| `environment.yml` | Conda environment for an x86-64 Linux machine (Python 3.11, PyTorch 2.7, Stable-Baselines3, ONNX, TensorRT 10, PyCUDA). It does not list `h5py`, which the harness and `utils/` import. It is not the Jetson Nano runtime environment: JetPack on the Nano ships TensorRT 8.x, and `nvbitfi_DNN_TRT.py` uses binding APIs that TensorRT 10 removed. |

## Pipeline

1. Train a DQN navigation policy (not included) and save it as `model_final.zip`.
2. Convert it: `python conv_from_zip_to_onnx.py --skip_trt`, then `python conv_from_onnx_to_trt.py --onnx model_final.onnx`. (Running `conv_from_zip_to_onnx.py` without `--skip_trt` does both steps.)
3. Run a fault-free reference pass and keep its output as the golden result.
4. Run the same layer with faults injected and compare against the golden output.

Step 4 is where the project stopped.

## Third-party code

`nvbitfi_DNN_TRT.py` is not fully my own work. It builds on an existing DNN fault-injection harness that I did not write, and the original repository belongs to someone else. Everything else in this repository is mine.

`Run_Layer.py` also imports a module named `nvbitfi_DNN` for the non-TensorRT path. That module is not included here.

## Running it

The scripts target a Jetson Nano with TensorRT. Paths in `run.sh` are set through the `APP_DIR` environment variable (default: the folder containing the script). The setup has not been re-tested since the project was paused.
