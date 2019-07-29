import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class CombinationModule(nn.Module):
    def __init__(self, in_size, out_size, cat_size):
        super(CombinationModule, self).__init__()
        self.up =  nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, stride=1),
                                 nn.ReLU(inplace=True))
        self.cat_conv =  nn.Sequential(nn.Conv2d(cat_size, out_size, kernel_size=1, stride=1),
                                       nn.ReLU(inplace=True))

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(F.interpolate(inputs2,inputs1.shape[2:],mode='bilinear', align_corners=False))
        return self.cat_conv(torch.cat((inputs1, outputs2), 1))


def make_skip_layers():
    layers = []
    layers += [CombinationModule(64, 64, 128)]
    layers += [CombinationModule(256, 64, 128)]
    layers += [CombinationModule(512, 256, 512)]
    layers += [CombinationModule(1024, 512, 1024)]
    return layers


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        num_kps = 5
        num_edges = 10

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.c0_conv = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(64, 64, 3, 1, 1),
                                     nn.ReLU(inplace=True))
        # segmentation modules============================================
        self.skip_combine = nn.ModuleList(make_skip_layers())
        self.seg_head = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 1, 3, 1, 1))
        # ================================================================
        # keypoint modules
        self.c4_up_conv = nn.Sequential(nn.Conv2d(1024, 512, 3, 1, 1), nn.ReLU(inplace=True))
        self.c3_up_conv = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU(inplace=True))
        self.c2_up_conv = nn.Sequential(nn.Conv2d(256, 64, 3,1, 1), nn.ReLU(inplace=True))
        self.c1_up_conv = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True))

        self.c3_cat_refine = nn.Sequential(nn.Conv2d(1024, 512, 1, 1), nn.ReLU(inplace=True))
        self.c2_cat_refine = nn.Sequential(nn.Conv2d(512, 256, 1, 1), nn.ReLU(inplace=True))
        self.c1_cat_refine = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.ReLU(inplace=True))
        self.c0_cat_refine = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.ReLU(inplace=True))


        self.kp_head_c3 = nn.Sequential(nn.Conv2d(512, 512, 7, 1, 3),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(512, num_kps, 7, 1, 3))

        self.short_offset_head_c3 = nn.Sequential(nn.Conv2d(512, 512, 7, 1, 3),
                                               nn.ReLU(inplace=True),
                                                  nn.Conv2d(512, 2*num_kps, 7, 1, 3))

        self.mid_offset_head_c3 = nn.Sequential(nn.Conv2d(512, 512, 7, 1, 3),
                                             nn.ReLU(inplace=True),
                                                nn.Conv2d(512, 4*num_edges, 7, 1, 3))


        self.kp_head_c2 = nn.Sequential(nn.Conv2d(256, 256, 7, 1, 3),
                                     nn.ReLU(inplace=True),
                                        nn.Conv2d(256, num_kps, 7, 1, 3))

        self.short_offset_head_c2 = nn.Sequential(nn.Conv2d(256, 256, 7, 1, 3),
                                               nn.ReLU(inplace=True),
                                                  nn.Conv2d(256, 2*num_kps, 7, 1, 3))

        self.mid_offset_head_c2 = nn.Sequential(nn.Conv2d(256, 256, 7, 1, 3),
                                             nn.ReLU(inplace=True),
                                                nn.Conv2d(256, 4*num_edges, 7, 1, 3))

        self.kp_head_c1 = nn.Sequential(nn.Conv2d(64, 64, 7, 1, 3),
                                     nn.ReLU(inplace=True),
                                        nn.Conv2d(64, num_kps, 7, 1, 3))

        self.short_offset_head_c1 = nn.Sequential(nn.Conv2d(64, 64, 7, 1, 3),
                                               nn.ReLU(inplace=True),
                                                  nn.Conv2d(64, 2*num_kps, 7, 1, 3))

        self.mid_offset_head_c1 = nn.Sequential(nn.Conv2d(64, 64, 7, 1, 3),
                                             nn.ReLU(inplace=True),
                                                nn.Conv2d(64, 4*num_edges, 7, 1, 3))


        self.kp_head_c0 = nn.Sequential(nn.Conv2d(64, 64, 7, 1, 3),
                                     nn.ReLU(inplace=True),
                                        nn.Conv2d(64, num_kps, 7, 1, 3))

        self.short_offset_head_c0 = nn.Sequential(nn.Conv2d(64, 64, 7, 1, 3),
                                               nn.ReLU(inplace=True),
                                                  nn.Conv2d(64, 2*num_kps, 7, 1, 3))

        self.mid_offset_head_c0 = nn.Sequential(nn.Conv2d(64, 64, 7, 1, 3),
                                             nn.ReLU(inplace=True),
                                                nn.Conv2d(64, 4*num_edges, 7, 1, 3))


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def get_patches(self, box, feat):
        y1, x1, y2, x2 = box
        _,h,w = feat.shape
        y1 = np.maximum(0, np.int32(np.round(y1 * h)))
        x1 = np.maximum(0, np.int32(np.round(x1 * w)))
        y2 = np.minimum(np.int32(np.round(y2 * h)), h - 1)
        x2 = np.minimum(np.int32(np.round(x2 * w)), w - 1)
        if y2<y1 or x2<x1 or y2-y1<2 or x2-x1<2:
            return None
        else:
            return (feat[:, y1:y2, x1:x2].unsqueeze(0))

    def mask_forward(self, i_x):
        pre = None
        for i in range(len(i_x)-1, -1, -1):
            if pre is None:
                pre = i_x[i]
            else:
                pre = self.skip_combine[i](i_x[i], pre)
                # if pre.shape[1] == 512:
                #     pre = self.crm(pre)
        return pre

    def forward(self, x, bboxes):
        dec0, dec1, dec2, dec3, feat_seg = self.forward_dec(x)
        seg = self.forward_seg(feat_seg, bboxes)
        return dec0, dec1, dec2, dec3, seg


    def forward_dec(self, x):
        c0 = self.c0_conv(x)  # c0: [2, 64, 512, 640]  [0, 257]

        x = self.conv1(x)
        x = self.bn1(x)
        c1 = self.relu(x)  # c1: [2, 64, 256, 320]  [0, 9.6]

        x = self.maxpool(c1)

        c2 = self.layer1(x)  # c2: [2, 256, 128, 160]   [0, 6]
        c3 = self.layer2(c2) # c3: [2, 512, 64, 80]     [0, 8]
        c4 = self.layer3(c3) # c4: [2, 1024, 32, 40]    [0, 3.4]

        c4_upsample = self.c4_up_conv(F.interpolate(c4, c3.size()[2:], mode='bilinear', align_corners=False))     # c4_upsample: [2, 512, 64, 80]  [0, 1,8]
        c3_cat = self.c3_cat_refine(torch.cat((c4_upsample, c3), 1))

        c3_upsample = self.c3_up_conv(F.interpolate(c3_cat, c2.size()[2:], mode='bilinear', align_corners=False)) # c3_upsample: [2, 256, 128, 160]  [0, 4.92]
        c2_cat = self.c2_cat_refine(torch.cat((c3_upsample, c2), 1))

        c2_upsample = self.c2_up_conv(F.interpolate(c2_cat, c1.size()[2:], mode='bilinear', align_corners=False)) # c2_upsample: [2, 64, 256, 320] [0, 13.4]
        c1_cat = self.c1_cat_refine(torch.cat((c2_upsample, c1), 1))

        c1_upsample = self.c1_up_conv(F.interpolate(c1_cat, c0.size()[2:], mode='bilinear', align_corners=False)) # c1_upsample: [2, 64, 512, 640] [0, 13,3]
        c0_cat = self.c0_cat_refine(torch.cat((c1_upsample, c0), 1))

        kp0 = torch.sigmoid(self.kp_head_c0(c0_cat))  # kp1: [2, 5, 512, 640]
        short0 = self.short_offset_head_c0(c0_cat)    # short1: [2, 10, 512, 640])
        mid0 = self.mid_offset_head_c0(c0_cat)        # mid1: [2, 40, 512, 640]


        kp1 = torch.sigmoid(self.kp_head_c1(c1_cat))  # kp1: [2, 5, 256, 320]
        short1 = self.short_offset_head_c1(c1_cat)    # short1: [2, 10, 256, 320])
        mid1 = self.mid_offset_head_c1(c1_cat)        # mid1: [2, 40, 256, 320]

        kp2 = torch.sigmoid(self.kp_head_c2(c2_cat))  # kp2: [2, 5, 128, 160]
        short2 = self.short_offset_head_c2(c2_cat)
        mid2 = self.mid_offset_head_c2(c2_cat)


        kp3 = torch.sigmoid(self.kp_head_c3(c3_cat))   # kp3: [2, 5, 64, 80]
        short3 = self.short_offset_head_c3(c3_cat)
        mid3 = self.mid_offset_head_c3(c3_cat)

        return [kp0,short0,mid0], [kp1,short1,mid1], [kp2,short2,mid2], [kp3,short3,mid3], [c0,c1,c2,c3,c4]


    def forward_seg(self, feat_seg, bboxes):
        mask_patches = [[] for i in range(len(bboxes))]
        mask_dets = [[] for i in range(len(bboxes))]

        # iterate batch
        for i in range(len(bboxes)):
            if len(bboxes[i])==0:
                continue
            for box, score in zip(bboxes[i][:, :4], bboxes[i][:, 4]):
                i_x = []
                y1, x1, y2, x2 = np.asarray(box,np.float32)
                h,w = feat_seg[0].shape[2:]
                for i_feat in feat_seg:
                    x = self.get_patches(box=[y1/float(h), x1/float(w),
                                              y2/float(h), x2/float(w)],
                                         feat=i_feat[i,:,:,:])
                    if x is not None:
                        i_x.append(x)
                    else:
                        break
                if len(i_x) == 0:
                    continue
                x = self.mask_forward(i_x)
                x = self.seg_head(x)
                x = torch.sigmoid(x)  # ranges from 0.5-1
                x = torch.squeeze(x, dim=0)
                x = torch.squeeze(x, dim=0)
                mask_patches[i].append(x)
                mask_dets[i].append(torch.Tensor(np.append(box,score)))
        return [mask_patches, mask_dets]


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
    return model
