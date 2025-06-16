from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_len, out_len, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_len, out_len, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_len)
        self.conv2 = nn.Conv2d(out_len, out_len, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_len)
        
        layers = []
        if stride != 1 or in_len != out_len:
            layers.append(nn.Conv2d(in_len, out_len, kernel_size=1, stride=stride, bias=False))
            layers.append(nn.BatchNorm2d(out_len))
            self.shortcut = nn.Sequential(*layers)

        else:
            self.shortcut = nn.Sequential()
        

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.bn2(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
    
class ResNet18Encoder(nn.Module):
    def __init__(self):
        super(ResNet18Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
                
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=1)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1)
        )
        
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1)
        )
        
        self.linear = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def create_block(self, out_len, stride):
        layers = []
        layers.append(ResidualBlock(self.in_len, out_len, stride=stride))
        self.in_len = out_len
        layers.append(ResidualBlock(self.in_len, out_len, stride=1))
        return nn.Sequential(*layers)
    
    def get_feature_space(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out
    
    def forward(self, x):
        out = self.get_feature_space(x)
        out = self.linear(out)
        return out