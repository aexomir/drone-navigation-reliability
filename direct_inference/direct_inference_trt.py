import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import h5py
import numpy as np

ENGINE_PATH = "../nvbitfi/model_final.trt"
INPUT_H5_PATH = "../nvbitfi/inputs.h5"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine, context, input_shapes):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        if engine.binding_is_input(binding):
            context.set_binding_shape(engine.get_binding_index(binding), input_shapes[binding])

        shape = tuple(context.get_binding_shape(engine.get_binding_index(binding)))
        size = trt.volume(shape)

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({"host": host_mem, "device": device_mem, "name": binding, "shape": shape})
        else:
            outputs.append({"host": host_mem, "device": device_mem, "name": binding, "shape": shape})

    return inputs, outputs, bindings, stream

def main():
    engine = load_engine(ENGINE_PATH)

    with engine.create_execution_context() as context:
        # Load input data
        with h5py.File(INPUT_H5_PATH, "r") as f:
            img_data = f["img"][:]
            vec_data = f["vec"][:]

        # Map binding names to shapes
        input_shapes = {}
        for binding in engine:
            if engine.binding_is_input(binding):
                if "img" in binding.lower():
                    input_shapes[binding] = img_data.shape
                elif "vec" in binding.lower():
                    input_shapes[binding] = vec_data.shape

        # Allocate buffers
        inputs, outputs, bindings, stream = allocate_buffers(engine, context, input_shapes)

        # Copy data into host buffers
        for inp in inputs:
            if "img" in inp["name"].lower():
                inp["host"][:] = img_data.ravel()
            elif "vec" in inp["name"].lower():
                inp["host"][:] = vec_data.ravel()

        # Copy inputs to device
        for inp in inputs:
            cuda.memcpy_htod_async(inp["device"], inp["host"], stream)

        # Run inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # Copy outputs to host
        for out in outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], stream)

        stream.synchronize()

        # Print outputs
        for out in outputs:
            output_array = np.array(out["host"]).reshape(out["shape"])
            print(f"Output {out['name']} shape {out['shape']}:\n{output_array}\n")

if __name__ == "__main__":
    main()
