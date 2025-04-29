import torch
import pickle
import os
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.optim import AdamW

# define CIFAR-100 dataset path
cifar100_dir = './cifar-100-python/'


# load CIFAR-100 dataset
def load_cifar100_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    return batch['data'], np.array(batch['fine_labels'])


def load_cifar100_train_data(cifar100_dir):
    # load train data
    train_data, train_labels = load_cifar100_batch(os.path.join(cifar100_dir, 'train'))

    return train_data, train_labels


# access CIFAR-10 data
X_train, y_train = load_cifar100_train_data(cifar100_dir)


# Preprocess the images: CIFAR-100 images are 32x32, but ViT expects 224x224 images
def preprocess_image(image):
    image = Image.fromarray(image)
    image = image.resize((224, 224))  # Resize to 224x224 for ViT
    return image


# create dataset class
class CIFAR100Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, processor):
        self.images = images
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = preprocess_image(self.images[idx].reshape(32, 32, 3))  # Ensure image is 32x32x3
        label = self.labels[idx]
        # Use the processor to convert the image to tensor format suitable for ViT
        inputs = self.processor(images=image, return_tensors="pt")

        # Ensure the correct shape for ViT: batch_size x channels x height x width
        inputs = {key: value.squeeze(0) for key, value in inputs.items()}  # Remove batch dimension

        inputs['labels'] = torch.tensor(label)
        return inputs


# Load the processor and model
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Change the classifier head to match CIFAR-10 classes (10 classes)
model.classifier = torch.nn.Linear(model.config.hidden_size, 100)


# continue training on pre-trained model
model.load_state_dict(torch.load('./trained_model.pth', weights_only=True))


# Fine-tuning for cifar-10
# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

# Set up optimizer and training parameters
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()
model.train()

# Create training DataLoader
train_dataset = CIFAR100Dataset(X_train, y_train, processor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
epochs = 1  # You can increase this for better accuracy
for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0
    model.train()

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
        inputs = {key: value.to(device) for key, value in batch.items() if key != 'labels'}
        labels = batch['labels'].to(device)

        # Forward
        outputs = model(**inputs)
        logits = outputs.logits
        loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {acc*100:.2f}% ")

    # save model
    if epoch == epochs - 1:
        torch.save(model.state_dict(), 'trained_model.pth')
