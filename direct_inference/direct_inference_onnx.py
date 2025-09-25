import onnx
import onnxruntime as ort
import numpy as np
import h5py

onnx_path = "../nvbitfi/model_final.onnx"
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
dataset_file = "../nvbitfi/inputs.h5"

with h5py.File(dataset_file, "r") as hf:
    Input_img = np.array(hf["img"], dtype=np.float32)
    Input_vec = np.array(hf["vec"], dtype=np.float32)

ort_sess = ort.InferenceSession(onnx_path)
# actions, values, log_prob = ort_sess.run(None, {'img': Input_img, 'vec': Input_vec})
result = ort_sess.run(None, {'img': Input_img, 'vec': Input_vec})

# print(actions, values, log_prob)
print(result)

# Check that the predictions are the same
# with th.no_grad():
#     print(model.policy(th.as_tensor(observation), deterministic=True))