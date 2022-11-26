#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Loading Models
"""

from __future__ import annotations

import unittest

from one.vision.classification.alexnet import AlexNet
from one.vision.classification.convnext import ConvNeXt
from one.vision.classification.densenet import DenseNet
from one.vision.classification.googlenet import GoogleNet
from one.vision.classification.inception import Inception3
from one.vision.classification.lenet import LeNet
from one.vision.classification.mobileone import MobileOne
from one.vision.classification.resnet import ResNet
from one.vision.classification.resnext import ResNeXt
from one.vision.classification.shufflenet_v2 import ShuffleNetV2
from one.vision.classification.squeezenet import SqueezeNet
from one.vision.classification.vgg import VGG
from one.vision.classification.wide_resnet import WideResNet
from one.vision.enhancement.dual_cnn import DualCNN
from one.vision.enhancement.ffanet import FFANet
from one.vision.enhancement.finet import FINet
from one.vision.enhancement.hinet import HINet
from one.vision.enhancement.mbllen import MBLLEN
from one.vision.enhancement.retinexnet import RetinexNet
from one.vision.enhancement.zeroadce import ZeroADCE
from one.vision.enhancement.zerodce import ZeroDCE
from one.vision.enhancement.zerodce_tiny import ZeroDCETiny
from one.vision.enhancement.zerodcepp import ZeroDCEPP
from one.vision.segmentation.unet import UNet


class TestModel(unittest.TestCase):
    
    def test_alexnet(self):
        model = AlexNet(num_classes=10, verbose=True)
        self.assertIsNotNone(model)
    
    def test_convnext_base(self):
        model = ConvNeXt(
            cfg         = "convnext-base",
            fullname    = "convnext-base",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_convnext_tiny(self):
        model = ConvNeXt(
            cfg         = "convnext-tiny",
            fullname    = "convnext-tiny",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
        
    def test_convnext_small(self):
        model = ConvNeXt(
            cfg         = "convnext-small",
            fullname    = "convnext-small",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_convnext_large(self):
        model = ConvNeXt(
            cfg         = "convnext-large",
            fullname    = "convnext-large",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
        
    def test_densenet121(self):
        model = DenseNet(
            cfg         = "densenet121",
            fullname    = "densenet121",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_densenet161(self):
        model = DenseNet(
            cfg         = "densenet161",
            fullname    = "densenet161",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_densenet169(self):
        model = DenseNet(
            cfg         = "densenet169",
            fullname    = "densenet169",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_densenet201(self):
        model = DenseNet(
            cfg         = "densenet201",
            fullname    = "densenet201",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_dual_cnn(self):
        model = DualCNN(verbose=True)
        self.assertIsNotNone(model)
        
    def test_ffanet(self):
        model = FFANet(verbose=True)
        self.assertIsNotNone(model)
    
    def test_finet(self):
        model = FINet(
            cfg         = "finet",
            fullname    = "finet",
            num_classes = 1000,
            pretrained  = False,
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_googlenet(self):
        model = GoogleNet(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(model)
    
    def test_hinet(self):
        model = HINet(
            cfg         = "hinet",
            fullname    = "hinet",
            num_classes = 1000,
            pretrained  = False,
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_inception3(self):
        model = Inception3(num_classes=1000, pretrained="imagenet", verbose=True)
        self.assertIsNotNone(model)
    
    def test_lenet(self):
        model = LeNet(num_classes=1000, verbose=True)
        self.assertIsNotNone(model)
    
    def test_mbllen(self):
        model = MBLLEN(verbose=True)
        self.assertIsNotNone(model)
    
    def test_mobileone_s0(self):
        model = MobileOne(
            cfg         = "mobileone-s0",
            fullname    = "mobileone-s0",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_mobileone_s1(self):
        model = MobileOne(
            cfg         = "mobileone-s1",
            fullname    = "mobileone-s1",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_mobileone_s2(self):
        model = MobileOne(
            cfg         = "mobileone-s2",
            fullname    = "mobileone-s2",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_mobileone_s3(self):
        model = MobileOne(
            cfg         = "mobileone-s3",
            fullname    = "mobileone-s3",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_mobileone_s4(self):
        model = MobileOne(
            cfg         = "mobileone-s4",
            fullname    = "mobileone-s4",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_mobileone_s0_unfused(self):
        model = MobileOne(
            cfg         = "mobileone-s0-unfused",
            fullname    = "mobileone-s0-unfused",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_mobileone_s1_unfused(self):
        model = MobileOne(
            cfg         = "mobileone-s1-unfused",
            fullname    = "mobileone-s1-unfused",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
        
    def test_mobileone_s2_unfused(self):
        model = MobileOne(
            cfg         = "mobileone-s2-unfused",
            fullname    = "mobileone-s2-unfused",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_mobileone_s3_unfused(self):
        model = MobileOne(
            cfg         = "mobileone-s3-unfused",
            fullname    = "mobileone-s3-unfused",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_mobileone_s4_unfused(self):
        model = MobileOne(
            cfg         = "mobileone-s4-unfused",
            fullname    = "mobileone-s4-unfused",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
        
    def test_resnet18(self):
        model = ResNet(
            cfg         = "resnet18",
            fullname    = "resnet18",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
        
    def test_resnet34(self):
        model = ResNet(
            cfg         = "resnet34",
            fullname    = "resnet34",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_resnet50(self):
        model = ResNet(
            cfg         = "resnet50",
            fullname    = "resnet50",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_resnet101(self):
        model = ResNet(
            cfg         = "resnet101",
            fullname    = "resnet101",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_resnet152(self):
        model = ResNet(
            cfg         = "resnet152",
            fullname    = "resnet152",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_resnext50_32x4d(self):
        model = ResNeXt(
            cfg         = "resnext50-32x4d",
            fullname    = "resnext50-32x4d",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)

    def test_resnext101_32x8d(self):
        model = ResNeXt(
            cfg         = "resnext101-32x8d",
            fullname    = "resnext101-32x8d",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_resnext101_64x4d(self):
        model = ResNeXt(
            cfg         = "resnext101-64x4d",
            fullname    = "resnext101-64x4d",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
        
    def test_retinexnet(self):
        model = RetinexNet(verbose=True)
        self.assertIsNotNone(model)
        
    def test_shufflenet_v2_x0_5(self):
        model = ShuffleNetV2(
            cfg         = "shufflenet-v2-x0.5",
            fullname    = "shufflenet-v2-x0.5",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_shufflenet_v2_x1_0(self):
        model = ShuffleNetV2(
            cfg         = "shufflenet-v2-x1.0",
            fullname    = "shufflenet-v2-x1.0",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_shufflenet_v2_x1_5(self):
        model = ShuffleNetV2(
            cfg         = "shufflenet-v2-x1.5",
            fullname    = "shufflenet-v2-x1.5",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_shufflenet_v2_x2_0(self):
        model = ShuffleNetV2(
            cfg         = "shufflenet-v2-x2.0",
            fullname    = "shufflenet-v2-x2.0",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
        
    def test_squeezenet_1_0(self):
        model = SqueezeNet(
            cfg         = "squeezenet-1.0",
            fullname    = "squeezenet-1.0",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_squeezenet_1_1(self):
        model = SqueezeNet(
            cfg         = "squeezenet-1.1",
            fullname    = "squeezenet-1.1",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_unet_32(self):
        model = UNet(cfg="unet-32.yaml", pretrained="lgg", verbose=True)
        self.assertIsNotNone(model)
    
    def test_vgg11(self):
        model = VGG(
            cfg         = "vgg11",
            fullname    = "vgg11",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
        
    def test_vgg11_bn(self):
        model = VGG(
            cfg         = "vgg11-bn",
            fullname    = "vgg11-bn",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_vgg13(self):
        model = VGG(
            cfg         = "vgg13",
            fullname    = "vgg13",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_vgg13_bn(self):
        model = VGG(
            cfg         = "vgg13-bn",
            fullname    = "vgg13-bn",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_vgg16(self):
        model = VGG(
            cfg         = "vgg16",
            fullname    = "vgg16",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_vgg16_bn(self):
        model = VGG(
            cfg         = "vgg16-bn",
            fullname    = "vgg16-bn",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)

    def test_vgg19(self):
        model = VGG(
            cfg         = "vgg19",
            fullname    = "vgg19",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)

    def test_vgg19_bn(self):
        model = VGG(
            cfg         = "vgg19-bn",
            fullname    = "vgg19-bn",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_wide_resnet50(self):
        model = WideResNet(
            cfg         = "wide-resnet50",
            fullname    = "wide-resnet50",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_wide_resnet101(self):
        model = WideResNet(
            cfg         = "wide-resnet101",
            fullname    = "wide-resnet101",
            num_classes = 1000,
            pretrained  = "imagenet",
            verbose     = True,
        )
        self.assertIsNotNone(model)
    
    def test_zeroadce(self):
        model = ZeroADCE(verbose=True)
        self.assertIsNotNone(model)
    
    def test_zerodce(self):
        model = ZeroDCE(pretrained="sice", verbose=True)
        self.assertIsNotNone(model)
    
    def test_zerodce_tiny(self):
        model = ZeroDCETiny(verbose=True)
        self.assertIsNotNone(model)
    
    def test_zerodcepp(self):
        model = ZeroDCEPP(pretrained="sice", verbose=True)
        self.assertIsNotNone(model)
