# source: https://github.com/irvinxav/Efficient-HetConv-Heterogeneous-Kernel-Based-Convolutions

class HetConv(nn.Module):
    def __init__(self, in_channels, out_channels, p):
        super(HetConv, self).__init__()
        # Groupwise Convolution
        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=p, bias=False)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.gwc(x) + self.pwc(x)
