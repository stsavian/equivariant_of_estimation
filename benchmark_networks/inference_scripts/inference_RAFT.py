import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
#from imageio import imsave

from networks.RAFT.core.raft import RAFT
from networks.RAFT.core.utils import flow_viz
from networks.RAFT.core.utils.utils import InputPadder
import sys

class inference(object):
    def __init__(self,args):
        self.DEVICE = 'cuda'
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model)) #stefano model[0]

        self.model = model.module
        self.model.to(self.DEVICE)
        self.model.eval()


    def __call__(self, inputs):
        img1 = inputs[0]
        img2 = inputs[1]
        with torch.no_grad():
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float().to(self.DEVICE)
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float().to(self.DEVICE)
            padder = InputPadder(img1.shape)
            img1 =img1[None,:,:,:]
            img2 = img2[None, :, :, :]
            image1, image2 = padder.pad(img1, img2)
            #inputs = [image1, image2]
            flow_low, flow_up = self.model(image1, image2, iters=20, test_mode=True)
            #flow_up = model(inputs)  # , iters=20, test_mode=True)
            flow = padder.unpad(flow_up).squeeze(0)
            flow_perm = flow.permute(1,0,2)#[0,:,:,:]
            flow_perm = flow_perm.permute(0, 2, 1)
            #padder = InputPadder(inputs[0].shape)
            #image1, image2 = padder.pad(inputs[0], inputs[1])

        return flow_perm.cpu().numpy()



