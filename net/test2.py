import torch.nn as nn
import torch
import torch.nn.functional as F
# net = nn.Sequential(
#     nn.Conv2d(1,64,kernel_size=(3,20),padding = (1,0),stride=1),
#     nn.BatchNorm2d(64),
#     nn.ReLU(),
# )
class Residual(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(Residual,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=(3,3),padding = (1,1),stride=stride)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=(3,3),padding = (1,1))
        self.conv3 = nn.Conv2d(in_channels,out_channels,kernel_size=(3,3),padding = (1,1),stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        y = F.leaky_relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        #print(y.shape)
        if self.conv3:
            x = self.conv3(x)
            #print(x.shape)
        return F.leaky_relu(x+y)

# blk = Residual(1,64,1)
# x = torch.rand((1,1,400,20))
# print(blk(x).shape)
class metal_net(nn.Module):
    def __init__(self,layers):
        super(metal_net, self).__init__()
        self.conv1 =nn.Conv2d(1,64,kernel_size=(3,3),padding = (1,1),stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.layer1 = self.resnet_block(64, 128, layers[0])
        self.layer2 = self.resnet_block(128, 256, layers[1])
        self.layer3 = self.last_block(256, 2)
        self.conv2 = nn.Conv2d(2,2, kernel_size=(1, 20), padding=(0, 0), stride=1)
        self.bn2 = nn.BatchNorm2d(2)
        self.relu = nn.LeakyReLU(inplace=True)
        self.gru=nn.GRU(input_size=400, hidden_size=10,num_layers=2,batch_first=True,bidirectional=2)
        self.softmax = nn.Softmax2d()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

    def resnet_block(self,in_chanels, out_channels, num_residuals, first_block=False):
        if first_block:
            assert in_chanels == out_channels
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(in_chanels, out_channels, stride=1))
            else:
                blk.append((Residual(out_channels, out_channels)))
        return nn.Sequential(*blk)

    def last_block(self,in_chanels, out_channels):
        return nn.Sequential(nn.Conv2d(in_chanels, out_channels, (1, 1)))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x= self.layer3(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        n,c,w,h = x.shape
        print(x.shape)
        x = torch.squeeze(x,3)
        x = self.gru(x)
        print(x.size())
        s_output = self.softmax(x)
        return x,s_output


def create_model():
    net = metal_net([2,4])
    return net

net = create_model()
x = torch.rand((1,1,400,20))
r1,r2 = net(x)
print(r1.shape)


# net.add_module('resnet_block1',resnet_block(64,64,2,first_block=True))
# net.add_module('resnet_block2',resnet_block(64,128,8))
# net.add_module('last_module',last_block(128,2))
# blk = Residual(1,64,1)
# x = torch.rand((1,1,400,20))
# for name,layer in net.named_children():
#     x = layer(x)
#     print(name,x.shape)
# y = net(x)
# print(y)