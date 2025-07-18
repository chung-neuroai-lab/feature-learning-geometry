'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut =nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride))#, bias=False),
             #   nn.BatchNorm2d(self.expansion*planes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, k, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64*k
        layers=[]
        layers+= [nn.Conv2d(3, 64*k, kernel_size=3, stride=1, padding=1),nn.ReLU()]
        #self.bn1 = nn.BatchNorm2d(64)
        a = self._make_layer(block, 64 * k, num_blocks[0], stride=1)
        layers+= [*a]
        a = self._make_layer(block, 128 * k, num_blocks[1], stride=2)
        layers+= [*a]
        a =self._make_layer(block, 256 * k, num_blocks[2], stride=2)
        layers+= [*a]
        a = self._make_layer(block, 512 * k, num_blocks[3], stride=2)
        layers+= [*a]
        layers += [nn.AvgPool2d(kernel_size=4)]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers += [block(self.in_planes, planes,   stride)]
            self.in_planes = planes * block.expansion
        return layers

    def forward(self, x):
        out = self.features(x)#F.relu(self.bn1(self.conv1(x)))
        #out = self.layer1(out)
        #out = self.layer2(out)
        #out = self.layer3(out)
        #out = self.layer4(out)
        #out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def ResNet18(k):
    return ResNet(BasicBlock, [2,2,2,2],k)


class VGG(nn.Module):
    def __init__(self, vgg_name,k):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name],k)
        self.classifier = nn.Linear(512*k, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg,k):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x*k, kernel_size=3, padding=1),
                          # nn.BatchNorm2d(x),
                           nn.ReLU(inplace=False)]
                in_channels = x*k
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
