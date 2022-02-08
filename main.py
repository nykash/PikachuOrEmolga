import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision import transforms
import tqdm


train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(128),
    transforms.ToTensor()
])


train_data = torchvision.datasets.ImageFolder('train', transform=train_transform)
train_data_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=16,
                                          shuffle=True)

test_data = torchvision.datasets.ImageFolder("test", transform=test_transform)
test_data_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=16,
                                          shuffle=True)


model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

lr = 0.001
mom = 0.9
epochs = 10

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom)
criterion = nn.CrossEntropyLoss()


def test():
    model.train(False)
    with torch.no_grad():
        running_loss = 0
        valid = 0
        for images, labels in tqdm.tqdm(test_data_loader):
            output = model(images)
            loss = criterion(output, labels)
            loss += running_loss
            _, predicted = torch.max(output.data, 1)
            valid += (predicted == labels).sum().item()

    running_loss/=len(test_data)

    print("\navg loss:", running_loss, ", accuracy:", str(100*(valid/len(test_data)))+"%")

#test()

for epoch in range(epochs):
    model.train()
    for i, (images, labels) in enumerate(train_data_loader):
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        if(i % 10 == 0):
            print("Epoch", epoch, ":", str(i*16)+"/"+str(len(train_data)), loss.item())
            torch.save(model.state_dict(), 'model_resnet.pth')
            torch.save(optimizer.state_dict(), 'optimizer_resnet.pth')

    test()


torch.save(model, "model.pth")

