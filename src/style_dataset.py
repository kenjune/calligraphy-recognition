import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import csv
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        # 做一个灰度
        transforms.RandomGrayscale(p=0.1)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])

    test_val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
    return train_transform, test_val_transform

def load_datasets(data_dir, train_transform, test_val_transform, val_ratio=0.2, test_ratio=0.1):
    dataset = datasets.ImageFolder(data_dir, transform=train_transform)

    total_len = len(dataset)
    val_len = int(total_len * val_ratio)
    test_len = int(total_len * test_ratio)
    print(f'Total samples:{total_len},Validation samples:{val_len},Test samples:{test_len}')
    train_len = total_len - val_len - test_len

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42))

    # Apply correct transforms for val/test
    val_dataset.dataset.transform = test_val_transform
    test_dataset.dataset.transform = test_val_transform
    class_to_idx=dataset.class_to_idx
    with open("class_mapping.csv","w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        for cls,idx in class_to_idx.items():
            writer.writerow([idx,cls])

    return train_dataset, val_dataset, test_dataset
