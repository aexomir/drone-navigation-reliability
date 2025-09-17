import os
import h5py
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

DEBUG = 1
# disable jetson nano watchdog: $ echo N > /sys/kernel/debug/gpu.0/timeouts_enabled
# enable jetson nano watchdog: $ echo Y > /sys/kernel/debug/gpu.0/timeouts_enabled

class TRT_load_embeddings:
    def __init__(self, path_dir, batch_size=1, layer_output_shape=(1,)) -> None:
        self.target_dtype = np.float32
        current_path = os.path.dirname(__file__)
        self.path_dir = os.path.join(current_path, path_dir)
        self.batch_size = batch_size
        self.layer_results = []

        self.onnx_model_name = "model_final.onnx"
        self.TRT_model_name = "model_final.trt"
        self.TRT_output_shape = layer_output_shape

        # --- Load input datasets (img + vec) ---
        dataset_file = os.path.join(self.path_dir, "inputs.h5")
        with h5py.File(dataset_file, "r") as hf:
            self.Input_img = np.array(hf["img"], dtype=np.float32)
            self.Input_vec = np.array(hf["vec"], dtype=np.float32)

        # --- Load TensorRT engine ---
        print(os.path.join(self.path_dir, self.TRT_model_name))
        with open(os.path.join(self.path_dir, self.TRT_model_name), "rb") as f:
            self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()


            # shape = self.context.get_tensor_shape(0)
            # print('num: '+self.engine.num_bindings)
            for i in range(self.engine.num_bindings):
                binding_name = self.engine.get_binding_name(i)
                binding_shape = self.engine.get_binding_shape(i)
                binding_is_input = self.engine.binding_is_input(i)
                if binding_is_input:
                    print(f"Input Binding: {binding_name}, Shape: {binding_shape}")
                else:
                    print(f"Output Binding: {binding_name}, Shape: {binding_shape}")

            #
            # Allocate host output
            self.output = np.empty(self.TRT_output_shape, dtype=self.target_dtype)
            print("output", self.output)
            if DEBUG: 
                print(self.TRT_output_shape)
                print(self.output.shape)

            # Prepare GPU buffers
            sample_img = self.Input_img[0:self.batch_size]
            sample_vec = self.Input_vec[0:self.batch_size]

            self.d_img = cuda.mem_alloc(sample_img.nbytes)
            # self.d_vec = cuda.mem_alloc(sample_vec.nbytes)
            self.d_output = cuda.mem_alloc(self.output.nbytes)

            # Order of bindings must match the engine input order
            self.bindings = [int(self.d_img),  int(self.d_output)]
            self.stream = cuda.Stream()

        if DEBUG:
            print("Num img samples:", len(self.Input_img))
            print("Num vec samples:", len(self.Input_vec))
            print("Shape img:", self.Input_img.shape)
            print("Shape vec:", self.Input_vec.shape)

    def __TRT_forward_function(self, img, vec):
        # Copy inputs to GPU
        cuda.memcpy_htod_async(self.d_img, img, self.stream)
        # cuda.memcpy_htod_async(self.d_vec, vec, self.stream)

        # Run inference
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)

        # Copy back output
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        self.stream.synchronize()

        return self.output

    def TRT_layer_inference(self):
        # Assume img and vec share the same first-dimension batch size
        max_batches = float(len(self.Input_img)) / float(self.batch_size)

        for batch in range(0, int(np.ceil(max_batches))):
            img = self.Input_img[batch*self.batch_size : (batch+1)*self.batch_size]
            vec = self.Input_vec[batch*self.batch_size : (batch+1)*self.batch_size]

            output = self.__TRT_forward_function(img, vec)
            self.layer_results.append(output)

        embeddings_outputs = np.concatenate(self.layer_results)


        if DEBUG:
            print("Final output shape:", embeddings_outputs.shape)

        log_path_file = os.path.join(self.path_dir, "outputs.h5")
        with h5py.File(log_path_file, "w") as hf:
            hf.create_dataset("layer_output", data=embeddings_outputs, compression="gzip")
