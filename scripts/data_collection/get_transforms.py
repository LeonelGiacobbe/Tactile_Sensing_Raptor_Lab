import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm # For a progress bar

def get_mean_std(dataset_path, image_size=224, batch_size=64):
    """
    Calculates the mean and standard deviation of an image dataset.

    Args:
        dataset_path (str): Path to the root directory of your image dataset.
                            Assumes a structure compatible with ImageFolder (e.g.,
                            dataset_path/class_name/image.jpg).
        image_size (int): The size to resize images to (e.g., 224 for ResNet).
        batch_size (int): Batch size for the DataLoader.

    Returns:
        tuple: A tuple containing:
            - mean (list): List of mean values for R, G, B channels.
            - std (list): List of standard deviation values for R, G, B channels.
    """

    # Define a transform that only converts to tensor and resizes
    # No normalization here, as we are calculating the stats for it
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor() # This automatically scales pixel values to [0, 1]
    ])

    # Create a dataset object (e.g., ImageFolder for classification datasets)
    # If your data is not organized into classes, you'll need a custom Dataset class
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    
    # Create a DataLoader to efficiently load images in batches
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=torch.multiprocessing.cpu_count(), # Use all CPU cores for faster loading
        shuffle=False # No need to shuffle for statistics calculation
    )

    channels_sum = torch.zeros(3)
    channels_sq_sum = torch.zeros(3)
    num_pixels = 0

    print("Calculating dataset mean and standard deviation...")
    for images, _ in tqdm(loader):
        # images shape: (batch_size, channels, height, width)
        # Reshape images to (batch_size, channels, height * width) and then
        # to (channels, batch_size * height * width) for easier calculation
        images = images.view(images.size(0), images.size(1), -1) # Flatten spatial dimensions
        
        # Sum over all pixels for each channel
        channels_sum += images.sum(dim=[0, 2]) # Sum across batch and flattened spatial dims
        
        # Sum of squares over all pixels for each channel
        channels_sq_sum += (images ** 2).sum(dim=[0, 2])
        
        # Count total pixels
        num_pixels += images.size(0) * images.size(2) # batch_size * (height * width)

    # Calculate mean and standard deviation
    mean = channels_sum / num_pixels
    std = torch.sqrt(channels_sq_sum / num_pixels - mean ** 2)

    return mean.tolist(), std.tolist()

# --- Example Usage ---
if __name__ == '__main__':
    # Replace 'path/to/your/training_dataset' with the actual path to your training data
    # This should be the root folder containing subfolders for each class.
    # For example:
    #   my_dataset/
    #   ├── class_A/
    #   │   ├── img1.png
    #   │   └── img2.png
    #   └── class_B/
    #       ├── img3.png
    #       └── img4.png
    
    # If your dataset is not structured this way, you'll need to adapt the Dataset loading.
    # For Gelsight images, if they are just raw images in a folder without class labels,
    # you'd use a custom Dataset that just loads images.
    
    # Example for Gelsight raw images in a flat folder:
    # import os
    # from PIL import Image
    # class GelsightDataset(torch.utils.data.Dataset):
    #     def __init__(self, root_dir, transform=None):
    #         self.root_dir = root_dir
    #         self.transform = transform
    #         self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    #
    #     def __len__(self):
    #         return len(self.image_files)
    #
    #     def __getitem__(self, idx):
    #         img_path = self.image_files[idx]
    #         image = Image.open(img_path).convert('RGB') # Ensure RGB
    #         if self.transform:
    #             image = self.transform(image)
    #         return image, 0 # Return a dummy label as there are no classes

    # Then in get_mean_std:
    # dataset = GelsightDataset(root_dir=dataset_path, transform=transform)

    training_data_path = 'dataset' # <--- IMPORTANT: Change this

    try:
        dataset_mean, dataset_std = get_mean_std(training_data_path)
        print(f"Calculated Mean: {dataset_mean}")
        print(f"Calculated Std: {dataset_std}")

        # Now you can use these values in your transform:
        # self.transform = transforms.Compose([
        #     transforms.Resize([res_size, res_size]),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=dataset_mean, std=dataset_std)
        # ])

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure 'training_data_path' is set correctly and the dataset structure is compatible with ImageFolder or your custom Dataset.")