import torch
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet
import torch.nn.functional as F
import math
from torch import nn

#
# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1,bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel,1, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16,epsilon=1e-5, mode='l2', after_relu=False):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.alpha = nn.Parameter(torch.ones(1, channel, 1, 1))
        # self.gamma = nn.Parameter(torch.zeros(1, channel, 1, 1))
        # self.beta = nn.Parameter(torch.zeros(1, channel, 1, 1))
        # self.epsilon = epsilon
        # self.mode = mode
        # self.after_relu = after_relu
        #
        # self.conv_mask = nn.Conv2d(channel, 1, kernel_size=1)  # context Modeling
        # self.softmax = nn.Softmax(dim=2)

        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=True)
            # nn.Sigmoid()
        )
        # self.conv2d = nn.Conv2d(channel // reduction, channel, 1, bias=True)
        self.hard_sigmoid=hard_sigmoid

    def forward(self, x):
        # if self.mode == 'l2':
        #     embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
        #

            # norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        b, c, _, _ = x.size()
        # batch, channel, height, width = x.size()
        # input_x = x
        # # [N, C, H * W]
        # input_x = input_x.view(batch, channel, height * width)
        # # [N, 1, C, H * W]
        # input_x = input_x.unsqueeze(1)
        # # [N, 1, H, W]
        # context_mask = self.conv_mask(x)
        # # [N, 1, H * W]
        # context_mask = context_mask.view(batch, 1, height * width)
        # # [N, 1, H * W]
        # context_mask = self.softmax(context_mask)  # softmax操作
        # # [N, 1, H * W, 1]
        # context_mask = context_mask.unsqueeze(3)
        # # [N, 1, C, 1]
        # context = torch.matmul(input_x, context_mask)
        # # [N, C, 1, 1]
        # context = context.view(batch, channel, 1, 1)
        y = self.avg_pool(x)
        y = self.fc(y)

        # y = self.conv2d(norm)
        # y2 = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        y = self.hard_sigmoid(y)
        return x * y.expand_as(x)

#


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')
            sys.exit()

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate






import torch
import torch.nn as nn

torch.set_default_tensor_type(torch.cuda.FloatTensor)


class SubSpace(nn.Module):
    """
    Subspace class.
    ...
    Attributes
    ----------
    nin : int
        number of input feature volume.
    Methods
    -------
    __init__(nin)
        initialize method.
    forward(x)
        forward pass.
    """

    def __init__(self, nin):
        super(SubSpace, self).__init__()
        self.conv_dws = nn.Conv2d(
            nin, nin, kernel_size=1, stride=1, padding=0, groups=nin
        )
        self.bn_dws = nn.BatchNorm2d(nin, momentum=0.9)
        self.relu_dws = nn.ReLU(inplace=False)
        self.se = SELayer(nin)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv_point = nn.Conv2d(
            nin, 1, kernel_size=1, stride=1, padding=0, groups=1
        )
        self.bn_point = nn.BatchNorm2d(1, momentum=0.9)
        self.relu_point = nn.ReLU(inplace=False)

        self.softmax = nn.Softmax(dim=2)

        # self.gct = GCT(nin)

    def forward(self, x):
        out = self.conv_dws(x)
        out = self.bn_dws(out)
        # out = self.relu_dws(out)
        out = self.se(out)
        out = self.maxpool(out)

        # avg_out = torch.mean(out, dim=1, keepdim=True)
        # max_out, _ = torch.max(out, dim=1, keepdim=True)
        # out = torch.cat([avg_out, max_out], dim=1)

        out = self.conv_point(out)
        out = self.bn_point(out)
        out = self.relu_point(out)

        m, n, p, q = out.shape
        out = self.softmax(out.view(m, n, -1))
        out = out.view(m, n, p, q)

        out = out.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        out = torch.mul(out, x)

        out = out + x
        # out = self.gct(out)
        return out


class ULSAM(nn.Module):
    """
    Grouped Attention Block having multiple (num_splits) Subspaces.
    ...
    Attributes
    ----------
    nin : int
        number of input feature volume.
    nout : int
        number of output feature maps
    h : int
        height of a input feature map
    w : int
        width of a input feature map
    num_splits : int
        number of subspaces
    Methods
    -------
    __init__(nin)
        initialize method.
    forward(x)
        forward pass.
    """

    def __init__(self, nin, nout, h, w, num_splits):
        super(ULSAM, self).__init__()

        assert nin % num_splits == 0

        self.nin = nin
        self.nout = nout
        self.h = h
        self.w = w
        self.num_splits = num_splits

        self.subspaces = nn.ModuleList(
            [SubSpace(int(self.nin / self.num_splits)) for i in range(self.num_splits)]
        )

    def forward(self, x):
        group_size = int(self.nin / self.num_splits)

        # split at batch dimension
        sub_feat = torch.chunk(x, self.num_splits, dim=1)

        out = []
        for idx, l in enumerate(self.subspaces):
            out.append(self.subspaces[idx](sub_feat[idx]))

        out = torch.cat(out, dim=1)

        return out


#
# class SEayer(nn.Module):
#     def __init__(self, channel, reduction=16,epsilon=1e-5, mode='l2', after_relu=False):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.alpha = nn.Parameter(torch.ones(1, channel, 1, 1))
#         self.gamma = nn.Parameter(torch.zeros(1, channel, 1, 1))
#         self.beta = nn.Parameter(torch.zeros(1, channel, 1, 1))
#         self.epsilon = epsilon
#         self.mode = mode
#         self.after_relu = after_relu
#         self.fc = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel, 1, bias=True),
#             # nn.Sigmoid()
#         )
#         self.hard_sigmoid=hard_sigmoid
#
#     def forward(self, x):
#         if self.mode == 'l2':
#             embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
#             # norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
#         # b, c, _, _ = x.size()
#         # y = self.avg_pool(x)
#         y = self.fc(embedding)
#         y = self.hard_sigmoid(y)
#         return x * y.expand_as(x)
#





# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16,epsilon=1e-5, mode='l2', after_relu=False):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.alpha = nn.Parameter(torch.ones(1, channel, 1, 1))
#         self.gamma = nn.Parameter(torch.zeros(1, channel, 1, 1))
#         self.beta = nn.Parameter(torch.zeros(1, channel, 1, 1))
#         self.epsilon = epsilon
#         self.mode = mode
#         self.after_relu = after_relu
#
#         self.conv_mask = nn.Conv2d(channel, 1, kernel_size=1)  # context Modeling
#         self.softmax = nn.Softmax(dim=2)
#
#         self.fc = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel, 1, bias=True)
#             # nn.Sigmoid()
#         )
#         # self.conv2d = nn.Conv2d(channel // reduction, channel, 1, bias=True)
#         self.hard_sigmoid=hard_sigmoid
#
#     def forward(self, x):
#         if self.mode == 'l2':
#             embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
#
#
#             # norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
#         b, c, _, _ = x.size()
#         # batch, channel, height, width = x.size()
#         # input_x = x
#         # # [N, C, H * W]
#         # input_x = input_x.view(batch, channel, height * width)
#         # # [N, 1, C, H * W]
#         # input_x = input_x.unsqueeze(1)
#         # # [N, 1, H, W]
#         # context_mask = self.conv_mask(x)
#         # # [N, 1, H * W]
#         # context_mask = context_mask.view(batch, 1, height * width)
#         # # [N, 1, H * W]
#         # context_mask = self.softmax(context_mask)  # softmax操作
#         # # [N, 1, H * W, 1]
#         # context_mask = context_mask.unsqueeze(3)
#         # # [N, 1, C, 1]
#         # context = torch.matmul(input_x, context_mask)
#         # # [N, C, 1, 1]
#         # context = context.view(batch, channel, 1, 1)
#         # y = self.avg_pool(x)
#         y = self.fc(embedding)
#
#         # y = self.conv2d(norm)
#         # y2 = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
#         y = self.hard_sigmoid(y)
#         return x * y.expand_as(x)
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         # self.hard_sigmoid = hard_sigmoid
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
#

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)





class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,  kernel_size=1, ratio=2, dw_size=3,downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride



        self.oup = planes
        init_channels = math.ceil(planes / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inplanes, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) ,
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        residual = x
        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)


        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)

        y1 = self.primary_conv(out)
        y2 = self.cheap_operation(y1)
        out = torch.cat([y1, y2], dim=1)

        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,kernel_size=1,dw_size=3,
                 base_width=64, dilation=1, norm_ayer=None,reduction=16):
        super(SEBottleneck, self).__init__()
        self.gct = GCT(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.ghost = GhostModule(planes, planes, relu=False)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        # self.se = SELayer(planes * 4, reduction)

        # def __init__(self, nin, nout, h, w, num_splits):
        self.sa = ULSAM(planes * 4,planes * 4,0,0,8)
        # self.sa1 = SpatialAttention()
        # self.gct = GCT(planes * 4)
        self.downsample = downsample
        self.stride = stride
        # #
        #
        # self.oup = planes
        # init_channels = math.ceil(planes / 2)
        # new_channels = init_channels * 1
        #
        # self.primary_conv = nn.Sequential(
        #     nn.Conv2d(inplanes, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
        #     nn.BatchNorm2d(init_channels),
        #     nn.ReLU(inplace=True),
        # )
        #
        # self.cheap_operation = nn.Sequential(
        #     nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
        #     nn.BatchNorm2d(new_channels),
        #     nn.ReLU(inplace=True),
        # )

    def forward(self, x):
        residual = x
        # out = self.gct(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.sa(out)
        # out = self.sa1(out)*out
        # out2 = self.sa(out)*out
        # out = self.gct(out)
        # out = out1+out2



        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def se_resnet18(num_classes =1_000):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet34(num_classes=1_000):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet50(num_classes=1000, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(
            "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"))
    return model


def se_resnet101(num_classes=1_000):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet152(num_classes=1_000):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model