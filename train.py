

from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
import warnings
import argparse
import torch.optim as optim
import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from inception_net import GoogLeNet

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr',  type=int, default=2e-4)
parser.add_argument('--use_cuda',  type=bool, default=True)

args = parser.parse_args()

transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])


if args.dataset == 'cifar10':
    train_data = CIFAR10(root='data/cifar10', train=True, download=True, transform=transform)
    test_data = CIFAR10(root='data/cifar10', train=False, download=True, transform=transform)

if args.dataset == 'cifar100':
    train_data = CIFAR100(root='data/cifar100', train=True, download=True, transform=transform)
    test_data = CIFAR100(root='data/cifar100', train=False, download=True, transform=transform)

use_cuda = args.use_cuda and torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else None
n_epochs = args.n_epochs
batch_size = args.batch_size
lr = args.lr

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = GoogLeNet()
if use_cuda:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Train
train_losses = []
# test_losses = []

for epoch in range(n_epochs):
    for (batch_x, batch_y) in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        print(batch_x.min())
        print(batch_x.max())
        logits = model(batch_x).unsqueeze(-1)
        probs = torch.softmax(logits, dim=1)
        loss = F.cross_entropy(probs, batch_y.unsqueeze(-1))
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        train_losses.append(loss.item())

    print(f'{epoch + 1}/{n_epochs} epochs | loss = {loss.item()}')


