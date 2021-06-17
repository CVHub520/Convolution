import torch
import torch.nn as nn

# source: https://github.com/bobo0810/HS-ResNet/blob/main/hsresnet/hs_block.py

class HSBlock(nn.Module):
    '''
    替代3x3卷积
    '''
    def __init__(self, in_ch, s=8):
        '''
        特征大小不改变
        :param in_ch: 输入通道
        :param s: 分组数
        '''
        super(HSBlock, self).__init__()
        self.s = s
        self.module_list = nn.ModuleList()

        in_ch_range=torch.Tensor(in_ch)
        in_ch_list = list(in_ch_range.chunk(chunks=self.s, dim=0))

        self.module_list.append(nn.Sequential())
        channel_nums = []
        for i in range(1,len(in_ch_list)):
            if i == 1:
                channels = len(in_ch_list[i])
            else:
                random_tensor = torch.Tensor(channel_nums[i-2])
                _, pre_ch = random_tensor.chunk(chunks=2, dim=0)
                channels= len(pre_ch)+len(in_ch_list[i])
            channel_nums.append(channels)
            self.module_list.append(self.conv_bn_relu(in_ch=channels, out_ch=channels))
        self.initialize_weights()

    def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        return conv_bn_relu

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = list(x.chunk(chunks=self.s, dim=1))
        for i in range(1, len(self.module_list)):
            y = self.module_list[i](x[i])
            if i == len(self.module_list) - 1:
                x[0] = torch.cat((x[0], y), 1)
            else:
                y1, y2 = y.chunk(chunks=2, dim=1)
                x[0] = torch.cat((x[0], y1), 1)
                x[i + 1] = torch.cat((x[i + 1], y2), 1)
        return x[0]
