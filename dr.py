import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import seaborn as sns

train_csv ='/home/interns/sreeja/dataset/train.csv'
train_image_dir ='/home/interns/sreeja/dataset/train_images'

#test_csv = '/content/drive/MyDrive/dataset/test.csv'
#test_image_dir ='/content/drive/MyDrive/dataset/test_images'

class DRDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, f"{self.dataframe.iloc[idx, 0]}.png")
        image = Image.open(img_name).convert('RGB')
        label = self.dataframe.iloc[idx, 1] if 'diagnosis' in self.dataframe.columns else -1

        if self.transform:
            image = self.transform(image)

        return image, label

trainLabels = pd.read_csv(train_csv)
print(trainLabels.head())

#testLabels = pd.read_csv(test_csv)
#print(testLabels.head())

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#train_df, val_df = train_test_split(trainLabels, train_size=0.8, shuffle=True, random_state=1)

train_val_df, test_df = train_test_split(trainLabels, train_size=0.8, shuffle=True, random_state=1)

train_df, val_df = train_test_split(train_val_df, train_size=0.9, shuffle=True, random_state=1)

print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")

train_dataset = DRDataset(train_df, train_image_dir, transform=data_transforms)
val_dataset = DRDataset(val_df, train_image_dir, transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#test_dataset = DRDataset(testLabels, test_image_dir, transform=data_transforms)
test_dataset = DRDataset(test_df, train_image_dir, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(trainLabels.columns)
print(trainLabels.head())

vc = train_df['diagnosis'].value_counts()
plt.figure(figsize=(9, 5))
sns.barplot(x=vc.index, y=vc, palette="rocket")
plt.title("Number of pictures of each category", fontsize=15)
plt.savefig("/home/interns/sreeja/dataset_distribution.png")  
#plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (f"running on:{device}")

model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  #Freeze pretrained layers

num_classes = trainLabels['diagnosis'].nunique()
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, num_classes)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

epochs = 10
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(epochs):
    #training
    model.train()
    running_loss, correct, total = 0, 0, 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_losses.append(running_loss / len(train_loader))
    train_accs.append(correct / total)

    #validation
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_losses.append(val_loss / len(val_loader))
    val_accs.append(correct / total)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, "
          f"Train Acc: {train_accs[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
          f"Val Acc: {val_accs[-1]:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.title("Loss")
plt.savefig("/home/interns/sreeja/loss_plot.png")  
#plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Validation Accuracy")
plt.legend()
plt.title("Accuracy")
plt.savefig("/home/interns/sreeja/accuracy_plot.png")  
#plt.show()

#testing

model.eval()
predictions,true_labels = [],[]
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

print("Predictions for test set:", predictions[:10])

print("True Labels:", true_labels[:10])

from sklearn.metrics import accuracy_score
test_accuracy = accuracy_score(true_labels, predictions)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

#saving model

print(model)

from torchsummary import summary
summary(model, input_size=(3, 224, 224))

model_save_path = "/home/interns/sreeja/dr_model.pth"
torch.save(model.state_dict(), model_save_path)
#torch.save(model, "/content/drive/MyDrive/full_model.pth")

print(f"Model saved to {model_save_path}")
