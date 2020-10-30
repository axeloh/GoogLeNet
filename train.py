from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
from torch.optim.lr_scheduler import StepLR
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
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--lr',  type=int, default=1e-3)
parser.add_argument('--gpu',  type=bool, default=True)
parser.add_argument('--modelname',  type=str, default='model')
parser.add_argument('--save_every',  type=int, default=5)
parser.add_argument('--lr_scheduler',  type=bool, default=True)

args = parser.parse_args()

transform = {
    'train': transforms.Compose([
            transforms.Resize((224, 224), interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
}


num_classes = 10
if args.dataset == 'cifar10':
    train_data = CIFAR10(root='data/cifar10', train=True, download=True, transform=transform['train'])
    test_data = CIFAR10(root='data/cifar10', train=False, download=True, transform=transform['test'])

elif args.dataset == 'cifar100':
    num_classes = 100
    train_data = CIFAR100(root='data/cifar100', train=True, download=True, transform=transform['train'])
    test_data = CIFAR100(root='data/cifar100', train=False, download=True, transform=transform['test'])


def poly_decay(n_epochs, epoch, init_lr, power=1.0):
    # initialize the maximum number of epochs, base learning rate,
    # and power of the polynomial
    maxEpochs = n_epochs
    baseLR = init_lr

    # compute the new learning rate based on polynomial decay
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

    # return the new learning rate
    return alpha


def compute_loss_acc(model, test_loader):
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


def save_acc_plot(taccs, vaccs, epoch, modelname):
    plt.plot([i for i in range(len(taccs))], taccs, label='train acc')
    plt.plot([i for i in range(len(taccs))], vaccs, label='val acc')
    plt.title(f'Accuracy during training [Epoch {epoch}]')
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.5, linestyle='--')
    plt.savefig(f'output/{modelname}_acc_plot', bbox_inches='tight')
    plt.clf()


def save_loss_plot(tlosses, vlosses, epoch, modelname):
    plt.plot([i for i in range(len(tlosses))], tlosses, label='train loss')
    plt.plot([i for i in range(len(tlosses))], vlosses, label='val loss')
    plt.title(f'Loss during training [Epoch {epoch}]')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.5, linestyle='--')
    plt.savefig(f'output/{modelname}_loss_plot', bbox_inches='tight')
    plt.clf()


use_cuda = args.gpu and torch.cuda.is_available()
print(f'Using cuda: {use_cuda}')
device = torch.device('cuda') if use_cuda else None
n_epochs = args.epochs
batch_size = args.bs
lr = args.lr

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = GoogLeNet(num_classes=num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
if args.lr_scheduler:
    scheduler = StepLR(optimizer, step_size=25, gamma=0.1)

# Train
train_losses = []
train_accs = []
val_losses = []
val_accs = []

init_loss, init_acc = compute_loss_acc(model, test_loader)
print(f'Init val loss: {init_loss:.3f}')
print(f'Init val accuracy: {init_acc:.3f}')

for epoch in range(n_epochs):
    for (batch_x, batch_y) in tqdm(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_x).unsqueeze(-1)
        optimizer.zero_grad()

        loss = F.cross_entropy(logits, batch_y.unsqueeze(-1))
        loss.backward()
        optimizer.step()

    if args.lr_scheduler:
        #scheduler.step()

        # Update learning rate
        new_lr = poly_decay(n_epochs, epoch, lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    # Calc train accuracy
    probs = torch.softmax(logits, dim=1)
    winners = probs.argmax(dim=1).squeeze()
    corrects = (winners == batch_y)
    train_acc = corrects.sum().float() / float(batch_y.size(0))
    train_accs.append(train_acc.item())
    train_losses.append(loss.item())

    # Calc val loss and accuracy
    val_loss, val_acc = compute_loss_acc(model, test_loader)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f'{epoch + 1}/{n_epochs} epochs | train_loss = {loss.item():.3f} | train_acc = {train_acc.item():.3f} | '
          f'val_loss = {val_loss:.3f} | val_acc = {val_acc:.3f}')

    # Save model
    if epoch % args.save_every == 0:
        torch.save(model.state_dict(), f'models/{args.modelname}_{args.dataset}_epoch{epoch}.pt')

    # Save plot
    save_acc_plot(train_accs, val_accs, epoch, args.modelname)
    save_loss_plot(train_losses, val_losses, epoch, args.modelname)


print(f'Training done.')




