import sys

import numpy as np
import time
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

# print('cuda version: ', torch.version.cuda)

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def get_dataloader_workers():
    """Use 4 processes to read the data.

    Defined in :numref:`sec_fashion_mnist`"""
    return 4

def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory.

    Defined in :numref:`sec_fashion_mnist`"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))
    
def load_cifar10(batch_size, resize=None):
    """Download the CIFAR-10 dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.CIFAR10(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.CIFAR10(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


class Residual(nn.Module):
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

sys.path.append('/home/geoalmtbs/vita/fastmoe/fmoe')
from resnetff import FMoEResNetFF
from resnetconv import FMoEResNetConv

class CustomizedMoEFF(FMoEResNetFF):
    def __init__(self, d_model, d_inner, d_output, dropout, pre_lnorm=False, moe_num_expert=64, moe_top_k=2):
        activation = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        super().__init__(num_expert=moe_num_expert, d_model=d_model, d_hidden=d_inner, d_output=d_output, top_k=moe_top_k,
                activation=activation)

        self.pre_lnorm = pre_lnorm
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = super().forward(self.layer_norm(inp))
            core_out = self.dropout(core_out)

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = super().forward(inp)
            core_out = self.dropout(core_out)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output

class CustomizedMoEResidual(FMoEResNetConv):
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels, d_model, moe_num_expert=8, moe_top_k=2,
                 use_1x1conv=False, strides=1):

        super().__init__(num_expert=moe_num_expert, num_channels=num_channels, d_model=d_model, top_k=moe_top_k)
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(super().forward(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

def resnet_block_moe(input_channels, num_channels, num_residuals, hw, num_expert,
                first_block=False):
    blk = []
    h, w = hw
    d_model = num_channels * h * w 
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(CustomizedMoEResidual(num_channels, num_channels, d_model, moe_num_expert=num_expert))
    return blk

def accuracy(y_hat, y):
    """Compute the number of correct predictions.

    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, axis=1)
    cmp = y_hat.type(y.dtype) == y
    # print(d2l)
    return float(torch.sum(cmp.type(y.dtype)))
# d2l = 

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train(net, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6).

    Defined in :numref:`sec_lenet`"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    # animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
    #                         legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            #     animator.add(epoch + (i + 1) / num_batches,
            #                  (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f'epoch {epoch:2d}, loss {train_l:.3f}, train acc {train_acc:.3f}, '
            f'test acc {test_acc:.3f}')
        # animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

def run():
    use_ff_moe, use_conv_moe_b3, use_conv_moe_b7 = False, False, False
    moe_top_k, num_expert = 2, 16
    lr, num_epochs, batch_size, resize = 0.05, 100, 2048, 32

    input_channels = 3
    # resize is used to control memory usage

    b1 = nn.Sequential(nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
                        nn.BatchNorm2d(64), nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))

    if use_conv_moe_b3 is True:
        b3 = nn.Sequential(*resnet_block_moe(64, 128, 2, (4, 4), num_expert))
    else: 
        b3 = nn.Sequential(*resnet_block(64, 128, 2))

    b4 = nn.Sequential(*resnet_block(128, 256, 2))

    # if use_ff_moe is True:
    #     b6 = CustomizedMoEFF(512, 256, 512, 0.5, pre_lnorm=False, moe_num_expert=8, moe_top_k=2)
    #     net = nn.Sequential(b1, b2, b3, b4, b5,
    #                         nn.AdaptiveAvgPool2d((1,1)),
    #                         nn.Flatten(), b6, nn.Linear(512, 10))
    # else:
    #     net = nn.Sequential(b1, b2, b3, b4, b5,
    #                         nn.AdaptiveAvgPool2d((1,1)),
    #                         nn.Flatten(), nn.Linear(512, 256),
    #                         nn.ReLU(), nn.Dropout(0.5),
    #                         nn.Linear(256, 512), nn.Linear(512, 10))

    if use_conv_moe_b7 is True:
        b5 = nn.Sequential(*resnet_block_moe(256, 512, 2, (1, 1), num_expert))
    else:
        b5 = nn.Sequential(*resnet_block(256, 512, 2))
        
    b6 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())

    if use_ff_moe is True:
        b7 = CustomizedMoEFF(512, 256, 512, 0.5, pre_lnorm=False, moe_num_expert=num_expert, moe_top_k=2)

    else:
        b7 = nn.Sequential(nn.Linear(512, 256),
                        nn.ReLU(), nn.Dropout(0.5),
                        nn.Linear(256, 512))

    net = nn.Sequential(b1, b2, b3, b4, b5, b6, b7, nn.Linear(512, 10))

    train_iter, test_iter = load_cifar10(batch_size, resize=resize)
    train(net, train_iter, test_iter, num_epochs, lr, try_gpu())

if __name__ == '__main__':
    run()
    
'''
to do
add argparser for convenience
add auto-logger
'''

