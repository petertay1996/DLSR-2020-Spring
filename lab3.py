from __future__ import division
import torch
import torch.nn as nn
#from torchvision import datasets ,models,transforms
from torchvision import datasets ,transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
from pathlib import Path
import torchvision.models as models
#from thop import profile
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

#Scalable Parameter
phi = 1
alpha = 1
beta = 1
gamma = 1

s_depth = alpha**phi
s_width = beta**phi
s_resolution = gamma**phi

print('alpha:{} * beta:{}^2 * gamma:{}^2 = {}'.format(alpha,beta,gamma,alpha*(beta**2.0)*(gamma**2.0)))

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=s_width*64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=s_width*64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    layer_v1 = [3, 4, 6, 3]
    layer_v2 = [int(n*s_depth) for n in layer_v1]
    return _resnet('resnet18', Bottleneck, layer_v2, pretrained, progress,**kwargs)

#import matplotlib.pyplot as plt

#To determine if your system supports CUDA
#print("==> Check devices..")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print("Current device: ",device)

#Also can print your current GPU id, and the number of GPUs you can use.
#print("Our selected device: ", torch.cuda.current_device())
#print(torch.cuda.device_count(), " GPUs is available")

TRAIN = "/usr/src/food11re/skewed_training"
VALID = "/usr/src/food11re/validation"
TEST = "/usr/src/food11re/evaluation"

# number of subprocesses to use for data loading
num_workers = 1
# how many samples per batch to load
batch_size = 32
# learning rate
LR = 0.001

# convert data to a normalized torch.FloatTensor
train_transforms = transforms.Compose([
    transforms.Resize((int(s_resolution*224),int(s_resolution*224))),
    transforms.ToTensor()
])

valid_transforms = transforms.Compose([
    transforms.Resize((int(s_resolution*224),int(s_resolution*224))),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((int(s_resolution*224),int(s_resolution*224))),
    transforms.ToTensor()
])

# choose the training and test datasets
train_data = datasets.ImageFolder(TRAIN, transform=train_transforms)
valid_data = datasets.ImageFolder(VALID,transform=valid_transforms)
test_data = datasets.ImageFolder(TEST, transform=test_transforms)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,  num_workers=num_workers,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,  num_workers=num_workers)

model = resnet18(pretrained = False)

import torch.optim as optim
from collections import OrderedDict

fc = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(2048,100)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(100,11)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.fc = fc

model = model.to(device)

#loss function
criterion = nn.CrossEntropyLoss()
#optimization algorithm
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# number of epochs to train the model
n_epochs = 30

layer = [n*s_depth for n in [3,4,6,3]]
print('Epoch:{}, layer:{}, width_per_group:{}, resolution:{}*{}'.format(n_epochs,layer,s_width*64,int(s_resolution*224),int(s_resolution*224)))
valid_loss_min = np.Inf # track change in validation loss

#train_losses,valid_losses=[],[]
for epoch in range(1, n_epochs+1):

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    correct = 0
    #print('running epoch: {}'.format(epoch))
    ###################
    # train the model #
    ###################
    model.train()
    for i, (data, target) in enumerate(train_loader,0):
        # move tensors to GPU if CUDA is available
        data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)

        _, pred = output.max(1)
        # if the model predicts the same results as the true
        # label, then the correct counter will plus 1
        correct += pred.eq(target).sum().item()
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # validate the model #
    ######################
    model.eval()
    for i,(data, target) in enumerate(valid_loader,0):
        # move tensors to GPU if CUDA is available
        data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
    
    # calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
    
    # print training/validation statistics
    #print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}\tTraining Accuracy: {:.6f}'.format(
    #    train_loss, valid_loss, 100.*correct/len(train_data)))
    end.record()
    torch.cuda.synchronize()

    print('Execution time: {}s, Memory Usage: {}MiB'.format(start.elapsed_time(end)*0.001,torch.cuda.max_memory_cached(device)/1048576.0))

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
    #    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
    #    valid_loss_min,
    #    valid_loss))
        torch.save(model.state_dict(), 'model.pt')
        valid_loss_min = valid_loss

def test(loaders, model, criterion):
    model.load_state_dict(torch.load('model.pt'))
    print("Lab 3-2:")
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.
    class_correct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    class_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    model.eval()
    correct_top3 = 0.
    for batch_idx, (data, target) in enumerate(loaders):
        # move to GPU
        data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        _, top3 = output.data.topk(max(1,3),1,True,True)
        pred = output.data.max(1, keepdim=True)[1]

        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

        for i in range(pred.shape[0]):
            pr = pred[i].item()
            gt = target.data[i].item()

            class_num[gt] += 1
            if pr==gt:
                class_correct[gt] += 1
            if pr==top3[i,0].item() or pr==top3[i,1].item() or pr==top3[i,2].item():
                correct_top3 += 1

    print('Test set: Top 1 Accuracy: %2d/%2d (%2d%%)' % (correct , total,100.* correct / total))

#    for i in range(11):
#        print('Class %3d: %4d/%3d  %5d%%' %(i,class_correct[i],class_num[i], 100.*class_correct[i]/class_num[i]))

model.cuda()
test(test_loader, model, criterion)


