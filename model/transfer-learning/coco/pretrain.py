import os
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Define paths
data_dir = 'coco/'
ann_file = os.path.join(data_dir, 'annotations/instances_train2014.json')
img_dir = os.path.join(data_dir, 'train2014')

# Initialize COCO API
coco = COCO(ann_file)

# Define the "bear" and "cow" classes
class_names = ['bear', 'cow']
class_ids = [coco.getCatIds(catNms=[name])[0] for name in class_names]

# Get image IDs for "bear" and "cow" classes
img_ids = []
for class_id in class_ids:
    img_ids.extend(coco.getImgIds(catIds=class_id))
img_ids = list(set(img_ids))  # Remove duplicates

# Custom Dataset for Bear and Cow Classes
class COCODataset(Dataset):
    def __init__(self, coco, img_dir, img_ids, class_ids, transform=None):
        self.coco = coco
        self.img_dir = img_dir
        self.img_ids = img_ids
        self.class_ids = class_ids
        self.transform = transform
        self.class_map = {class_id: i for i, class_id in enumerate(class_ids)}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        # Get labels (either 0 for "bear" or 1 for "cow")
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.class_ids, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        label = self.class_map[anns[0]['category_id']]  # Assume one label per image

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create dataset and data loader
dataset = COCODataset(coco, img_dir, img_ids, class_ids, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load a pre-trained model
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: bear and cow

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 500
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader):.4f}")

    model_save_path = f"resnet50_model_coco_pretrained.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

print("Finished Training")
