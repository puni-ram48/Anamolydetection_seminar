import os
from enum import Enum
import PIL
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

def visualize_resized_images(dataset, num_samples=5, save_path="resized_images_visualization.png"):
    """
    Visualize original and resized images side by side
    """
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        if i >= len(dataset):
            break
            
        # Get original image path
        img_path, _ = dataset.data_to_iterate[i]
        
        # Load and display original image
        original_img = PIL.Image.open(img_path)
        original_img_rgb = original_img.convert('RGB') if original_img.mode != 'RGB' else original_img
        
        axes[i, 0].imshow(original_img_rgb)
        axes[i, 0].set_title(f"Original\n{os.path.basename(img_path)}\nSize: {original_img.size}")
        axes[i, 0].axis('off')
        
        # Load and display transformed image
        data_item = dataset[i]
        transformed_img = data_item["image"]
        
        # Denormalize the image
        transformed_img_np = transformed_img.numpy().transpose(1, 2, 0)
        transformed_img_np = transformed_img_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        transformed_img_np = np.clip(transformed_img_np, 0, 1)
        
        axes[i, 1].imshow(transformed_img_np)
        axes[i, 1].set_title(f"Resized & Transformed\nSize: {transformed_img.shape[1:]} (HxW)")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.show()

def convert_to_rgb(img):
    """Convert grayscale to RGB"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def identity_transform(x):
    """Identity transformation (replaces lambda x: x)"""
    return x

class ChestXRayDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Chest X-Ray Pneumonia dataset.
    """
    
    def __init__(
        self,
        source,
        classname="NORMAL",
        resize=256,
        imagesize=256,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        rotate_degrees=0,
        translate=0,
        brightness_factor=0,
        contrast_factor=0,
        saturation_factor=0,
        gray_p=0,
        h_flip_p=0,
        v_flip_p=0,
        scale=0,
        augment=False,
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.classname = classname
        self.train_val_split = train_val_split
        self.augment = augment
        self.data_to_iterate = self.get_image_data()
        
        # Transformations
        transform_list = []
        
        # Convert grayscale to RGB using a named function
        transform_list.append(transforms.Lambda(convert_to_rgb))
        
        if resize is not None:
            transform_list.append(transforms.Resize(resize))
        
        if self.split == DatasetSplit.TRAIN and augment:
            if h_flip_p > 0:
                transform_list.append(transforms.RandomHorizontalFlip(h_flip_p))
            if brightness_factor > 0 or contrast_factor > 0:
                transform_list.append(transforms.ColorJitter(
                    brightness=brightness_factor, 
                    contrast=contrast_factor
                ))
        
        transform_list.append(transforms.CenterCrop(imagesize))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        
        self.transform_img = transforms.Compose(transform_list)

        # Mask transformation - simpler
        mask_transforms = []
        if resize is not None:
            mask_transforms.append(transforms.Resize(resize))
        mask_transforms.append(transforms.CenterCrop(imagesize))
        mask_transforms.append(transforms.ToTensor())
        
        self.transform_mask = transforms.Compose(mask_transforms)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        img_path, is_anomaly = self.data_to_iterate[idx]
        
        try:
            image = PIL.Image.open(img_path)
            image = self.transform_img(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = torch.zeros(self.imagesize)
        
        mask = torch.zeros([1, *image.size()[1:]])
        
        return {
            "image": image,
            "mask": mask,
            "classname": self.classname,
            "anomaly": "pneumonia" if is_anomaly else "normal",
            "is_anomaly": int(is_anomaly),
            "image_name": os.path.basename(img_path),
            "image_path": img_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        data_to_iterate = []
        
        normal_dir = os.path.join(self.source, self.split.value, "NORMAL")
        pneumonia_dir = os.path.join(self.source, self.split.value, "PNEUMONIA")
        
        # TRAINING: Only use normal samples
        if self.split == DatasetSplit.TRAIN:
            if os.path.exists(normal_dir):
                for img_file in os.listdir(normal_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        data_to_iterate.append((os.path.join(normal_dir, img_file), 0))
        
        # TESTING/VALIDATION: Use both normal and pneumonia
        elif self.split in [DatasetSplit.TEST, DatasetSplit.VAL]:
            # Normal samples
            if os.path.exists(normal_dir):
                for img_file in os.listdir(normal_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        data_to_iterate.append((os.path.join(normal_dir, img_file), 0))
            
            # Pneumonia samples (anomalies)
            if os.path.exists(pneumonia_dir):
                for img_file in os.listdir(pneumonia_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        data_to_iterate.append((os.path.join(pneumonia_dir, img_file), 1))
        
        print(f"Chest X-Ray {self.split.value}: Loaded {len(data_to_iterate)} images")
        return data_to_iterate
