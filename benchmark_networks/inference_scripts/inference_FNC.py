import numpy as np
import os
from imageio import imread, imwrite
import matplotlib.pylab as plt
import cv2


import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import sys
#sys.path.insert(0, '/python_scripts/FlowNetPytorch/')

#sys.path.append('../../')
#sys.path.append('../')
import networks.FlowNetPytorch.flow_transforms as flow_transforms
import networks.FlowNetPytorch.models as models

class inference(object):
    def __init__(self, train_pth):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.checkpoint = torch.load(train_pth)
        if 'div_flow' in self.checkpoint.keys():
            self.div_flow = self.checkpoint['div_flow']

        self.model = models.flownetc(self.checkpoint)  #
        self.model.cuda()
        self.model.eval()
        cudnn.benchmark = True
        self.input_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
            transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])])

    def __call__(self, inputs):
        img1 = self.input_transform(inputs[0].copy())
        img2 = self.input_transform(inputs[1].copy())
        with torch.no_grad():
            input_var = torch.cat([img1, img2]).unsqueeze(0)
            input_var = input_var.to(self.device)
            # print(type(input_var))
            output = self.model(input_var)
            # print(type(output))
            output = F.interpolate(output, size=img1.size()[-2:], mode='bilinear', align_corners=False)
            est_flo = (self.div_flow * output[0, :, :, :]).cpu().numpy().transpose(1, 2, 0)
            del input_var
        return est_flo



