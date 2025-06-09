import numpy as np

# Load the .npy file
data = np.load('../data_collection/dataset/gel.npy', allow_pickle=True)

# Print the contents
print(data)
