import torch
import pickle
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification
from tqdm import tqdm


# define CIFAR-10 dataset path
cifar10_dir = './cifar-10-batches-py/'


# load CIFAR-10 test data
def load_cifar10_test_data(cifar10_dir):
    with open(os.path.join(cifar10_dir, 'test_batch'), 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    data = batch['data']
    labels = batch['labels']
    return data, np.array(labels)


# Preprocess image
def preprocess_image(image, transform=None):
    image = Image.fromarray(image)
    if transform:
        image = transform(image)
    else:
        image = image.resize((224, 224))

    return image

# Dataset
class CIFAR10TestDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, processor, transform=None):
        self.images = images
        self.labels = labels
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(32, 32, 3)
        image = preprocess_image(image, self.transform)
        label = self.labels[idx]

        # Use the processor to convert the image to tensor format suitable for ViT
        inputs = self.processor(images=image, return_tensors="pt")

        inputs = {key: value.squeeze(0) for key, value in inputs.items()}

        return inputs, torch.tensor(label).long()

# Load model and processor
model = ViTForImageClassification.from_pretrained('./trained_model', num_labels=10)
processor = ViTImageProcessor.from_pretrained('./trained_model')


# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
model.eval()


# Transform
eval_transform = transforms.Resize((224, 224))


# Load test set
X_test, y_test = load_cifar10_test_data(cifar10_dir)
test_dataset = CIFAR10TestDataset(X_test, y_test, processor, transform=eval_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Loss function
criterion = torch.nn.CrossEntropyLoss()

# Evaluation
correct = 0
total = 0
total_loss = 0

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Evaluating"):
        inputs = {key: value.to(device) for key, value in inputs.items()}
        labels = labels.to(device)

        outputs = model(**inputs)
        logits = outputs.logits

        loss = criterion(logits, labels)
        total_loss += loss.item()

        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

# Results
avg_loss = total_loss / len(test_loader)
accuracy = correct / total

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Average Loss: {avg_loss:.4f}")
