import torch.nn as nn
import torch

##########################################################################
##---------- Dual input Unet----------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # nn.Sequential():Newrual Network Moudule will be added to compute orderly
        # 定义双向卷积的操作
        self.conv = nn.Sequential(
            # nn.Conv2d(self,in_channels,out_channels,kernel_size,stride,padding)
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


# UNet is to finish four downsample and four upsample
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # kernel_size(int or tuple) - max pooling的窗口大小
        # stride(int or tuple, optional) - max pooling的窗口移动的步长
        self.conv1 = DoubleConv(in_channels, 64)
        self.pooling1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = DoubleConv(64, 128)
        self.pooling2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = DoubleConv(128, 256)
        self.pooling3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = DoubleConv(256, 512)
        self.pooling4 = nn.MaxPool2d(2, stride=2)
        self.conv5 = DoubleConv(512, 1024)
        # inverse_conv
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_channels, kernel_size=1)

    # 前向遍历一遍UNet进行输出
    # forward函数的任务需要把输入层、网络层、输出层链接起来，实现信息的前向传导
    def forward(self, x):
        # dropout = nn.Dropout(p=0.2)
        E_block1 = self.conv1(x)
        # print(E_block1.shape)
        pooling1 = self.pooling1(E_block1)
        # print(pooling1.shape)
        E_block2 = self.conv2(pooling1)
        # print(E_block2.shape)
        pooling2 = self.pooling2(E_block2)
        # print(pooling2.shape)
        E_block3 = self.conv3(pooling2)
        # print(E_block3.shape)
        pooling3 = self.pooling3(E_block3)
        E_block4 = self.conv4(pooling3)
        # print(E_block4.shape)
        # E_block4 = dropout(E_block4)
        pooling4 = self.pooling4(E_block4)

        bottleneck = self.conv5(pooling4)
        # print("++++++++++++++++++++++++")
        # print(bottleneck.shape)
        # bottleneck = dropout(bottleneck)
        up6 = self.up6(bottleneck)
        # print("++++++++++++++++++++++++")
        # print(E_block4.shape)
        # print(up6.shape)
        # 在上采样阶段实现UNet的跳接结构，也就是UNet图的copy and crop

        merge6 = torch.cat([E_block4, up6], dim=1)
        D_block1 = self.conv6(merge6)
        up7 = self.up7(D_block1)

        # print("++++++++++++++++++++++++")
        # print(up7.shape)
        merge7 = torch.cat([E_block3, up7], dim=1)
        D_block2 = self.conv7(merge7)
        up8 = self.up8(D_block2)

        # print("++++++++++++++++++++++++")
        # print(E_block2.shape)
        # print(up8.shape)
        merge8 = torch.cat([E_block2, up8], dim=1)
        D_block3 = self.conv8(merge8)
        up9 = self.up9(D_block3)

        # print("++++++++++++++++++++++++")
        # print(E_block1.shape)
        # print(up9.shape)
        merge9 = torch.cat([E_block1, up9], dim=1)
        D_block4 = self.conv9(merge9)
        output = self.conv10(D_block4)
        # print("output", output.shape)
        # return nn.Sigmoid()(output)  # 将输出划定在0~1的区间
        return output