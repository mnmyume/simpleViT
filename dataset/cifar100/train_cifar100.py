import torch
import pickle
import os
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.optim import AdamW
from torchvision import transforms

# data augmentation transform
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomRotation(10),
])

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
def preprocess_image(image, transform=None):
    image = Image.fromarray(image)
    if transform:
        image = transform(image)
    else:
        image = image.resize((224, 224))  # Resize to 224x224 for ViT

    return image


# create dataset class
class CIFAR100Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, processor, transform=None):
        self.images = images
        self.labels = labels
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(32, 32, 3)  # Ensure image is 32x32x3
        image = preprocess_image(image, self.transform)

        label = self.labels[idx]
        # Use the processor to convert the image to tensor format suitable for ViT
        inputs = self.processor(images=image, return_tensors="pt")

        # Ensure the correct shape for ViT: batch_size x channels x height x width
        inputs = {key: value.squeeze(0) for key, value in inputs.items()}  # Remove batch dimension

        inputs['labels'] = torch.tensor(label)
        return inputs


# load raw config and set dropout
config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
config.hidden_dropout_prob = 0.1
config.attention_probs_dropout_prob = 0.1


# Load the processor and model
resume_training = True
if resume_training:
    model = ViTForImageClassification.from_pretrained('./trained_model', num_labels=100)
    processor = ViTImageProcessor.from_pretrained('./trained_model')
    print("Resuming training from ./trained_model")
else:
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    print("Starting fresh training from scratch")


# Change the classifier head to match CIFAR-100 classes (100 classes)
model.classifier = torch.nn.Linear(model.config.hidden_size, 100)


# Fine-tuning for cifar-10
# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

# Set up optimizer and training parameters
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss()
model.train()

# Create training DataLoader
train_dataset = CIFAR100Dataset(X_train, y_train, processor, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
epochs = 5
best_acc = 0
epochs_without_improvement = 0
patience = 1
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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


    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    acc = correct / total
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {acc*100:.2f}% ")

    # save model
    if epoch == epochs - 1 or acc > best_acc:
        best_acc = acc
        model.save_pretrained('./trained_model')
        processor.save_pretrained('./trained_model')
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement > patience:
        print(f"Early stopping after {epoch + 1} epochs.")
        break
