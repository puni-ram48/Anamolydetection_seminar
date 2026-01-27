import sys
sys.path.append(".")

from datasets.chest_xray import ChestXRayDataset, DatasetSplit, visualize_resized_images

# Initialize dataset
dataset = ChestXRayDataset(
    source="chest_xray",  # Update this path
    split=DatasetSplit.TRAIN,
    resize=256,
    imagesize=224,
    augment=False
)

# Visualize 5 samples
visualize_resized_images(dataset, num_samples=5, save_path="resized_images_comparison.png")

# Also visualize test samples
test_dataset = ChestXRayDataset(
    source="chest_xray",  # Update this path
    split=DatasetSplit.TEST,
    resize=256,
    imagesize=224,
    augment=False
)

visualize_resized_images(test_dataset, num_samples=5, save_path="test_resized_images_comparison.png")
