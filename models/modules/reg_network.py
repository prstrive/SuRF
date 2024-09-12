import torch.nn as nn
import torchsparse.nn as spnn


class BasicSparseConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        out = self.net(x)
        return out


class BasicSparseDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class SparseCostRegNet(nn.Module):
    """
    Sparse cost regularization network;
    require sparse tensors as input
    """

    def __init__(self, d_in, d_out=8, d_base=8):
        super(SparseCostRegNet, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_base = d_base

        self.conv0 = BasicSparseConvolutionBlock(d_in, d_base)

        self.conv1 = BasicSparseConvolutionBlock(d_base, d_base*2, stride=2)
        self.conv2 = BasicSparseConvolutionBlock(d_base*2, d_base*2)

        self.conv3 = BasicSparseConvolutionBlock(d_base*2, d_base*4, stride=2)
        self.conv4 = BasicSparseConvolutionBlock(d_base*4, d_base*4)

        self.conv5 = BasicSparseConvolutionBlock(d_base*4, d_base*8, stride=2)
        self.conv6 = BasicSparseConvolutionBlock(d_base*8, d_base*8)

        self.conv7 = BasicSparseDeconvolutionBlock(d_base*8, d_base*4, ks=3, stride=2)

        self.conv9 = BasicSparseDeconvolutionBlock(d_base*4, d_base*2, ks=3, stride=2)

        self.conv11 = BasicSparseDeconvolutionBlock(d_base*2, d_base, ks=3, stride=2)
        
        self.out_lin = nn.Linear(d_base, d_out, bias=False)

    def forward(self, x):
        """
        :param x: sparse tensor
        :return: sparse tensor
        """
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))

        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        
        out = self.out_lin(x.F)
        
        return out, x.F
    
    
class SparseCostRegNetList(nn.Module):
    def __init__(self, confs):
        super(SparseCostRegNetList, self).__init__()
        
        d_in = confs.get_list("d_in")
        d_out = confs.get_list("d_out")
        d_base = confs.get_list("d_base")
        self.num_stages = len(d_in)
        
        self.nets = nn.ModuleList()
        
        for i in range(self.num_stages):
            self.nets.append(SparseCostRegNet(d_in[i], d_out[i], d_base[i]))
        
    def forward(self, ipts, stage_idx):
        out, mid_feat = self.nets[stage_idx](ipts)
        return out, mid_feat