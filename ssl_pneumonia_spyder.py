import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from PIL import Image
import os

# ==========================
# DATASET PATH
# ==========================
DATA_PATH = "dataset/chest_xray"

# ==========================
# SIMCLR AUGMENTATION
# ==========================
class SimCLRTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.4,0.4,0.4,0.1),
            transforms.ToTensor()
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

# ==========================
# SSL DATASET
# ==========================
ssl_dataset = ImageFolder(
    root=os.path.join(DATA_PATH,"train"),
    transform=SimCLRTransform()
)

ssl_loader = DataLoader(ssl_dataset, batch_size=32, shuffle=True)

# ==========================
# SIMCLR MODEL
# ==========================
class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet18(weights=None)
        self.encoder.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128)
        )

    def forward(self,x):
        h = self.encoder(x)
        z = self.projector(h)
        return z

# ==========================
# CONTRASTIVE LOSS
# ==========================
def nt_xent_loss(z1,z2,temp=0.5):

    z1 = nn.functional.normalize(z1,dim=1)
    z2 = nn.functional.normalize(z2,dim=1)

    rep = torch.cat([z1,z2],dim=0)
    sim = torch.matmul(rep,rep.T)

    batch = z1.shape[0]

    labels = torch.cat([torch.arange(batch) for _ in range(2)],dim=0)
    labels = (labels.unsqueeze(0)==labels.unsqueeze(1)).float()

    mask = torch.eye(labels.shape[0],dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0],-1)
    sim = sim[~mask].view(sim.shape[0],-1)

    pos = sim[labels.bool()].view(labels.shape[0],-1)
    neg = sim[~labels.bool()].view(sim.shape[0],-1)

    logits = torch.cat([pos,neg],dim=1)
    labels = torch.zeros(logits.shape[0],dtype=torch.long).to(logits.device)

    logits = logits/temp

    return nn.CrossEntropyLoss()(logits,labels)

# ==========================
# TRAIN SELF SUPERVISED
# ==========================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimCLR().to(device)
optimizer = optim.Adam(model.parameters(),lr=1e-3)

SSL_EPOCHS = 100

for epoch in range(SSL_EPOCHS):

    total_loss = 0

    for (x1,x2),_ in ssl_loader:

        x1,x2 = x1.to(device),x2.to(device)

        z1 = model(x1)
        z2 = model(x2)

        loss = nt_xent_loss(z1,z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("SSL Epoch:",epoch+1,"Loss:",total_loss/len(ssl_loader))

# ==========================
# CLASSIFIER MODEL
# ==========================
class Classifier(nn.Module):
    def __init__(self,encoder):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(512,2)

    def forward(self,x):
        f = self.encoder(x)
        return self.fc(f)

classifier = Classifier(model.encoder).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(),lr=1e-4)

# ==========================
# FINETUNE DATA
# ==========================
train_dataset = ImageFolder(
    root=os.path.join(DATA_PATH,"train"),
    transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)

FT_EPOCHS = 100

for epoch in range(FT_EPOCHS):

    total_loss = 0

    for img,label in train_loader:

        img,label = img.to(device),label.to(device)

        out = classifier(img)
        loss = criterion(out,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("FineTune Epoch:",epoch+1,"Loss:",total_loss/len(train_loader))

# ==========================
# TEST ACCURACY
# ==========================
test_dataset = ImageFolder(
    root=os.path.join(DATA_PATH,"test"),
    transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
)

test_loader = DataLoader(test_dataset,batch_size=32)

correct = 0
total = 0

classifier.eval()

with torch.no_grad():

    for img,label in test_loader:

        img,label = img.to(device),label.to(device)

        out = classifier(img)
        _,pred = torch.max(out,1)

        total += label.size(0)
        correct += (pred==label).sum().item()

print("Test Accuracy:",100*correct/total)

