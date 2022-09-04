import torch.nn as nn
from torchsummary import summary
from thop import profile
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# train_set = datasets.MNIST('', train=True, download=True,
#                        transform=transforms.Compose([
#                            transforms.ToTensor()
#                        ]))

data_aug = transforms.Compose([
transforms.ToTensor(),
transforms.RandomRotation(15, expand=False),
])

train_set = datasets.MNIST('', train=True, download=True,
                       transform=data_aug
                       )

test_set = datasets.MNIST('', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

class FPGANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=3, stride=1, padding=0)
        self.linear1 = nn.Linear(1000,200)
        self.linear2 = nn.Linear(200, 10)
        self.flatten = nn.Flatten()
        self.weight_init()

    def weight_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.constant_(layer.bias, 0)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
    
        return x

def train(train_loader, net, criterion, optim, train_loss, train_acc, device, max_epoch=100):
    net.train()
    net.to(device)
    for epoch in range(0, max_epoch):
        correct = 0
        total = 0
        accuracy = 0
        loss_all = 0
        for idx, (data, label) in tqdm(enumerate(train_loader), desc=f"Training Epoch {epoch}"):
            data = data.to(device).half()
            label = label.to(device)

            optim.zero_grad()
            # print(torch.dtype(data))
            pred = net(data)
            loss = criterion(pred, label)
            loss.backward()
            optim.step()

            correct += torch.eq(torch.argmax(pred,dim=1), label).sum().float().item()
            loss_all += loss.cpu().item()
        train_loss.append(loss_all)
        total = len(train_loader.dataset)
        accuracy = correct/total
        train_acc.append(accuracy)
        print('Epoch: {}, loss: {:.3f}, acc: {:.3f}'.format(epoch+1,loss.item(), accuracy))

        torch.save(net, './{}/Acc{:.3f}_Epoch{}.pth'.format(root_save_dir, accuracy, epoch))
        test(test_loader, net, criterion, test_loss, test_acc, device)

def test(test_loader, net, criterion, test_loss, test_acc, device):
    net.eval()
    net.to(device)
    correct = 0
    total = 0
    accuracy = 0
    loss_all = 0
    for idx, (data, label) in tqdm(enumerate(test_loader), desc="Testing"):
        data = data.to(device).half()
        label = label.to(device)
        pred = net(data)
        loss = criterion(pred, label)
        loss_all += loss.cpu().item()
        correct += torch.eq(torch.argmax(pred,dim=1), label).sum().float().item()

    total = len(test_loader.dataset)
    accuracy = correct/total
    test_acc.append(accuracy)
    test_loss.append(loss_all)
    print('Test loss: {:.3f}, acc: {:.3f}'.format(loss.item(), accuracy))


if __name__ == "__main__":

    ### Hyper Params ###
    _batch_size = 128
    lr = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}, bz: {_batch_size}, lr: {lr}")
    ### Dataloader ###
    train_loader = DataLoader(train_set, batch_size=_batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=_batch_size, shuffle=False)

    net = FPGANet().cuda().half()
    # model = net
    # input = torch.randn(1, 1, 28, 28).cuda().half()
    # flops, params = profile(model, inputs=(input, ))
    # print("%s ------- params: %.2f Byte------- flops: %.2f" % (model, params, flops)) 
    # summary(net.cuda(), input_size=(1,28,28), batch_size=2, device="cuda") 

    optim = torch.optim.Adam(net.parameters(), lr = lr, eps=1e-3)
    criterion = nn.CrossEntropyLoss()
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    root_save_dir = "./float16"
    train(train_loader, net, criterion, optim, train_loss, train_acc, device, max_epoch=20)
    torch.save(train_loss, f"./{root_save_dir}/train_loss.txt")
    torch.save(train_acc, f"./{root_save_dir}/train_acc.txt")
    torch.save(test_loss, f"./{root_save_dir}/test_loss.txt")
    torch.save(test_acc, f"./{root_save_dir}/test_acc.txt")
    
