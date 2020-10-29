

from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
import warnings
import argparse
import torch.optim as optim
import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
from inception_net import GoogLeNet
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
parser.add_argument('--n_epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr',  type=int, default=1e-3)
parser.add_argument('--use_cuda',  type=bool, default=True)

args = parser.parse_args()

transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])


num_classes = 10
if args.dataset == 'cifar10':
    train_data = CIFAR10(root='data/cifar10', train=True, download=True, transform=transform)
    test_data = CIFAR10(root='data/cifar10', train=False, download=True, transform=transform)

if args.dataset == 'cifar100':
    num_classes = 100
    train_data = CIFAR100(root='data/cifar100', train=True, download=True, transform=transform)
    test_data = CIFAR100(root='data/cifar100', train=False, download=True, transform=transform)


def compute_val_loss_acc(model, test_loader):
    model.eval()
    loss = 0
    accs = []
    with torch.no_grad():
        for (batch_x, batch_y) in tqdm(test_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x).unsqueeze(-1)
            loss += F.cross_entropy(logits, batch_y.unsqueeze(-1)).item()

            probs = torch.softmax(logits, dim=1)
            winners = probs.argmax(dim=1).squeeze()
            corrects = (winners == batch_y)
            accuracy = corrects.sum().float() / float(batch_y.size(0))
            accs.append(accuracy)
    mean_acc = torch.mean(torch.tensor(accs))
    mean_loss = loss / len(test_loader)
    model.train()

    return mean_loss, mean_acc


use_cuda = args.use_cuda and torch.cuda.is_available()
print(f'Using cuda: {use_cuda}')
device = torch.device('cuda') if use_cuda else None
n_epochs = args.n_epochs
batch_size = args.batch_size
lr = args.lr

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = GoogLeNet(num_classes=num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Train
train_losses = []
train_accs = []
val_losses = []
val_accs = []

init_loss, init_acc = compute_val_loss_acc(model, test_loader)
print(f'Init val loss: {init_loss:.3f}')
print(f'Init val accuracy: {init_acc:.3f}')

for epoch in range(n_epochs):
    for (batch_x, batch_y) in tqdm(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_x).unsqueeze(-1)
        # probs = torch.softmax(logits, dim=1)  # F.cross_entropy expects logits
        optimizer.zero_grad()

        loss = F.cross_entropy(logits, batch_y.unsqueeze(-1))
        loss.backward()
        optimizer.step()

    # Calc train accuracy
    probs = torch.softmax(logits, dim=1)
    winners = probs.argmax(dim=1).squeeze()
    corrects = (winners == batch_y)
    train_acc = corrects.sum().float() / float(batch_y.size(0))
    train_accs.append(train_acc)
    train_losses.append(loss.item())

    # Calc val loss and accuracy
    val_loss, val_acc = compute_val_loss_acc(model, test_loader)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f'{epoch + 1}/{n_epochs} epochs | train_loss = {loss:.3f} | train_acc = {train_acc:.3f} | val_loss = {val_loss:.3f} | val_acc = {val_acc:.3f}')

# Plot
plt.plot([i for i in range(len(train_losses))], train_losses, label='train loss')
plt.plot([i for i in range(len(train_losses))], train_accs, label='train acc')
plt.plot([i for i in range(len(train_losses))], val_losses, label='val loss')
plt.plot([i for i in range(len(train_losses))], val_accs, label='val acc')
plt.title('Loss and Accuracy during training')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.savefig('output/loss_accuracy_plot')




