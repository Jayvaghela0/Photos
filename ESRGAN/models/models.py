import torch.nn as nn

class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(RRDBNet, self).__init__()
        self.model = self.make_network(in_nc, out_nc, nf, nb, gc)

    def make_network(self, in_nc, out_nc, nf, nb, gc):
        # Define the RRDB network architecture
        layers = []
        for _ in range(nb):
            layers += [ResidualDenseBlock(nf, gc)]
        layers += [nn.Conv2d(nf, nf, 3, 1, 1, bias=True)]
        layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        layers += [nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ResidualDenseBlock(nn.Module):
    def __init__(self, nf, gc):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x
