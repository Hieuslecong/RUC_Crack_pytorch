# dataset crop할때 unet paper와 다르게 진행  mirroring padding을 변형
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        if stride != 1 or in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.skip_bn = nn.BatchNorm2d(out_channels)
        else:
            self.skip_conv = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.skip_conv is not None:
            identity = self.skip_conv(identity)
            identity = self.skip_bn(identity)
        out += identity
        out = self.relu(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, stride=2, reduction=16):
        super().__init__()
        self.basic_block1 = BasicBlock(in_channels, out_channels, stride)
        self.basic_block2 = BasicBlock(out_channels, out_channels, 1)
        self.blocks = nn.Sequential(self.basic_block1, self.basic_block2, *([BasicBlock(out_channels, out_channels)] * (num_blocks - 1)))
        # self.scse = SCSE(out_channels, reduction)
        # self.add_scse=add_scse

    def forward(self, x):
        out = self.blocks(x)
        # if self.add_scse==True:
        #     out = self.scse(out)
        return out

class SCSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return module_input * x.expand_as(module_input)

class RUCNet(nn.Module):
    def __init__(self,num_classes=2, reduction=16):
        super().__init__()
        # Encoder downsample path
        self.scse_64 = SCSE(64, reduction)
        self.scse_128 = SCSE(128, reduction)
        self.scse_256 = SCSE(256, reduction)

        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.res_block1 = ResidualBlock(64, 64, 2, stride=2, reduction=reduction)
        self.res_block2 = ResidualBlock(64, 128, 2, stride=2, reduction=reduction)
        self.res_block3 = ResidualBlock(128, 256, 2, stride=2, reduction=reduction)
        self.res_block4 = ResidualBlock(256, 512, 2, stride=2, reduction=reduction)
        #self.down_sample=nn.MaxPool2d(kernel_size=2, stride=2)
        #self.res_block4 = ResidualBlock(512, 512, 2, stride=2, reduction=reduction)
        # Decoder upsample path
        # self.up_conv1 = nn.Sequential(
        #     nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=False)
        # )
        # self.up_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=False)
        # self.bn2 = nn.BatchNorm2d(256)
        # self.up_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.up_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False)
        # self.bn4 = nn.BatchNorm2d(64)
        # self.final_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)
        # upsample path
        self.upconv4 = nn.ConvTranspose2d(512, 256,kernel_size=2, stride=2, padding=0)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0)

        

        # Residual connections
        self.residual1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.residual2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.residual3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.residual4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Classifier
        self.conv_final = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)# 64x64
        #x1_d=self.down_sample(x1)
        #print('done layer1')
        x1 = self.res_block1(x)# 64x64
        x_sc64=self.scse_64(x1)
        #x2_d=self.down_sample(x2)
        #print('done layer2')

        x2 = self.res_block2(x1)# 128x128
        x_sc128=self.scse_128(x2)
        #x3_d=self.down_sample(x3)
        #print('done layer3')

        x3 = self.res_block3(x2)# 256x256
        x_sc256=self.scse_256(x3)
        #x4_d=self.down_sample(x4)
        #print('done layer4')
        x4 = self.res_block4(x3)# 256x256
        #x4_d=self.down_sample(x4)
        
        #x5 = self.res_block4(x4) # 512x512
        # Decoder
        d4 = self.upconv4(x4) #512x512
        d4 = torch.cat([d4, x_sc256], dim=1) #256+512x256
        d4 = self.residual1(d4) #256x256
        #d4 = self.res_block_up4(d4)#256x256
        d4=self.scse_256(d4)

        d3 = self.upconv3(d4) #256x256
        d3 = torch.cat([d3, x_sc128], dim=1) #128+256x256
        d3 = self.residual2(d3) #128x128
        #d3 = self.res_block_up3(d3)#128x128
        d3=self.scse_128(d3)
        d2 = self.upconv2(d3) #128x128
        d2 = torch.cat([d2, x_sc64], dim=1) #128+64x64
        d2 = self.residual3(d2) #64x64
        #d2 = self.res_block_up2(d2)#64x64
        d2=self.scse_64(d2)
        d1 = self.upconv1(d2) #64x64
        d1 = torch.cat([d1, x], dim=1) #64+64x64
        d1 = self.residual4(d1) #64x64
        #d1 = self.res_block_up1(d1)#64x64
        d1=self.scse_64(d1)
        x=self.conv_final(d1)# Classifier
        return x
    
# test
def test():
    x = torch.randn((3, 1, 161, 161))  # batch size, channel,height,width
    model = RUCNet(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
