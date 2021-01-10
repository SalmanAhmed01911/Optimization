#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -q efficientnet_pytorch')


# In[2]:


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[4]:


# Hyper-parameters
num_epochs = 20
learning_rate = 0.001


# In[5]:


transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])


# In[6]:


train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True, 
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                            train=False, 
                                            transform=transforms.ToTensor())


# In[7]:


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=64, 
                                          shuffle=False)





# In[11]:


class EfficientNetB6(nn.Module):
    '''
    Last fully connected layer changed to ouput EMBEDDINGS_SIZE-dim vector.
    '''
    
    def __init__(self, classes = 10):
        super(EfficientNetB6, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b6')
        self.model._fc = nn.Linear(in_features=self.model._fc.in_features, out_features=classes, bias=True)

    def forward(self, image):
        features = self.model(image)
        return features


# In[12]:


model = EfficientNetB6().cuda()
model.half()


# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[ ]:


# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    running_loss = 0.0
    running_corrects = 0
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_corrects += torch.sum(preds == labels.data)
        running_loss += loss.item() * images.size(0)

#         if (i+1) % 100 == 0:
#             print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
#                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double().item() / len(train_loader.dataset)
    print ("Epoch : " + str(epoch+1) + " Loss : " + str(epoch_loss) + " Acc : " + str(epoch_acc))

    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))


# In[ ]:




