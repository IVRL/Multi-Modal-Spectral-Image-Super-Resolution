import torch
import torch.nn as nn

EPS = 1e-3

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                     padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=True)
    
    def forward(self, x):
        residual = x
        
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, channels=14):
        super(ResNet, self).__init__()
        self.conv_first = nn.Conv2d(channels, 64, kernel_size=3, stride=1,
                padding=1, bias=True)
        self.conv_last = nn.Conv2d(64, channels, kernel_size=1, stride=1,
                padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        self.residual_part = self.make_residual(5)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        
    def make_residual(self, count):
        layers = []
        for _ in range(count):
            layers.append(ResBlock(64, 64))
        return nn.Sequential(*layers)
    
    def forward(self, x2):
        out = self.relu(self.conv_first(x2))
        out = self.residual_part(out)
        out = self.conv_last(out)
        #out = self.sigmoid(out)
        return out

class SecondResidualNet(nn.Module):
    def __init__(self, channels=14):
        super(SecondResidualNet, self).__init__()
        self.residual_layer = self.make_layer(7)
        
        self.input = nn.Conv2d(in_channels=channels+3, out_channels=64,
                kernel_size=3, stride=1, padding=1, bias=False)
        
        self.output = nn.Conv2d(in_channels=64, out_channels=channels,
                kernel_size=3, stride=1, padding=1, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                
    def make_layer(self, count):
        layers = []
        for _ in range(count):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64,
                kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x2, y2):
        residual = x2
        out = self.relu(self.input(torch.cat((x2,y2), 1)))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return out 
        
class ResidualLearningNet(nn.Module):
    def __init__(self, channels=14):
        super(ResidualLearningNet, self).__init__()
        self.residual_layer = self.make_layer(10)
        
        self.input = nn.Conv2d(in_channels=channels, out_channels=64,
                kernel_size=3, stride=1, padding=1, bias=False)
        
        self.output = nn.Conv2d(in_channels=64, out_channels=channels,
                kernel_size=3, stride=1, padding=1, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                
    def make_layer(self, count):
        layers = []
        for _ in range(count):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64,
                kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x2):
        residual = x2
        out = self.relu(self.input(x2))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return out
    
class MRAE(nn.Module):
    def __init__(self):
        super(MRAE, self).__init__()
        
    def forward(self, output, target, mask=None):
        relative_diff = torch.abs(output - target) / (target + 1.0/65535.0)
        if mask is not None:
            relative_diff = mask * relative_diff
        return torch.mean(relative_diff)

class SID(nn.Module):
    def __init__(self):
        super(SID, self).__init__()
        
    def forward(self, output, target, mask=None):
        
        output = torch.clamp(output, 0, 1)
        
        a1 = output * torch.log10((output + EPS) / (target + EPS))
        a2 = target * torch.log10((target + EPS) / (output + EPS))
        
        if mask is not None:
            a1 = a1 * mask
            a2 = a2 * mask
        
        a1_sum = a1.sum(dim=3).sum(dim=2)
        a2_sum = a2.sum(dim=3).sum(dim=2)
        
        errors = torch.abs(a1_sum + a2_sum)
        
        return torch.mean(errors)
                                                                      
                                                                      
                                                                      
                                                                      
                                                                      
                                                                      
                                                                      
                                                                      
                                                                      
                                                                      
                                                                      
                                                                      
                                                                      
                                                                      
