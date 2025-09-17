import h5py
import sys
import numpy as np

if len(sys.argv) != 4:
    print("Usage: python3 merge_h5.py output.h5 inputs_img.h5 inputs_vec.h5")
    sys.exit(1)

output_file = sys.argv[1]
img_file = sys.argv[2]
vec_file = sys.argv[3]

with h5py.File(output_file, "w") as hf_out:
    # load img and convert to float32
    with h5py.File(img_file, "r") as hf_img:
        img_data = np.array(hf_img["img"], dtype=np.float32)
        hf_out.create_dataset("img", data=img_data, compression="gzip")
        print(f"Added 'img' from {img_file} with shape {img_data.shape} and dtype {img_data.dtype}")

    # load vec and convert to float32
    with h5py.File(vec_file, "r") as hf_vec:
        vec_data = np.array(hf_vec["vec"], dtype=np.float32)
        hf_out.create_dataset("vec", data=vec_data, compression="gzip")
        print(f"Added 'vec' from {vec_file} with shape {vec_data.shape} and dtype {vec_data.dtype}")

print(f"Created merged file {output_file}")
