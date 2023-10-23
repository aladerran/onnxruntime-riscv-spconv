import numpy as np
import torch

def load_data_to_tensor_with_numpy(filename, shape):
    data = np.loadtxt(filename, delimiter=',', dtype=np.int32)
    reshaped_data = data.reshape(-1, 4)
    tensor = torch.tensor(reshaped_data)
    return tensor.reshape(shape)

# Example of usage:
# Let's say your tensor shape is (25, 4) based on the data you provided (100 total values, 4 per row)
tensor = load_data_to_tensor_with_numpy("input_coords.csv", (20, 4))

print(tensor)
