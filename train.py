import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
from torch import nn, optim
from tqdm import tqdm

# 1. Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Data Preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
train_dataset = ImageFolder('data/train', transform=transform)
val_dataset = ImageFolder('data/validation', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Model Definition
model = models.resnet18(pretrained=True)  # Load pre-trained ResNet
model.fc = nn.Linear(model.fc.in_features, 2)  # Adjust final layer for binary classification
model = model.to(DEVICE)

# 4. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5. Training Loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    correct = 0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {correct/len(train_dataset):.4f}")

# 6. Save Model
torch.save(model.state_dict(), 'models/resnet18.pth')
