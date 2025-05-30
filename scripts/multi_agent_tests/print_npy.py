import numpy as np

# Load the .npy file
data = np.load('../data_collection/dataset/wood_block.npy', allow_pickle=True)

# Print the contents
print(data)
