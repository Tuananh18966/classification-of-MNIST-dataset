import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Load MNIST

tranform = transforms.Compose([transforms.ToTensor(),])

datasets = torchvision.datasets.MNIST(root = './data', train = True, transform = tranform, download = False)

batch_size = 32
dataloader = torch.utils.data.DataLoader(dataset = datasets, batch_size = batch_size, shuffle = True, num_workers = 1)
test_loader = torch.utils.data.DataLoader(dataset = datasets, batch_size= batch_size, shuffle = True, num_workers= 1)
#Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 32, 3 ), 
                                     nn.ReLU(), 
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(32, 64, 3), 
                                     nn.ReLU(), 
                                     nn.MaxPool2d(2))
        self.fc = nn.Sequential(nn.Linear(64*5*5, 128 ), 
                                nn.ReLU(), 
                                nn.Linear(128, 10))
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        y_pred = self.fc(x)
        return y_pred
    
model = CNN().to(device)

# loss and optimizer 
learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# train loop
batch_total = len(dataloader)
print(batch_total)
num_epochs = 5
for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(dataloader):
        images = images.view(batch_size, 1, 28, 28).to(device)
        labels = labels.to(device)
        y_predicted = model(images)
        loss = criterion(y_predicted, labels)

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        if i%100==0:
            print(f'epoch: {epoch+1}/{num_epochs}, step: {i+1}/{batch_total}, loss: {loss.item():.3f}')


# test
n_correct = 0
with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(batch_size, 1, 28, 28).to(device)
            labels = labels.to(device)

            y_test = model(images)
            _, labels_pred = torch.max(y_test, dim=1)

            n_correct += (labels_pred == labels).sum().item()
acc = (n_correct/len(datasets))*100
print(f'accuracy: {acc:.3f} %')
torch.save(model.state_dict(), "mnist_cnn.pth")