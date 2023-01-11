import torch
from torchvision import transforms as vision_transforms

import numpy as np
import scipy.misc
#from scipy import ndimage
from imageio import imread, imsave
import os
from networks.irr.models.IRR_PWC import PWCNet as irr_PWCNet
#from networks.irr.models.IRR_PWC import PWCNet
#from networks.irr.models.pwcnet_irr import PWCNet  UNCOMMENT THIS AND CHECKPOINT FOR DIFFERENT MODELS
from networks.irr.utils.flow import flow_to_png_middlebury
import cv2



# ## Forward pass
# input_dict = {'input1' : im1_tensor, 'input2' : im2_tensor}
# output_dict = model.forward(input_dict)
#
# flow = output_dict['flow'].squeeze(0).detach().cpu().numpy()


class inference(object):
    def __init__(self,model_checkpoint_path,model_name='irr-PWC_Net'):
        # self.DEVICE = 'cuda'
        # model = torch.nn.DataParallel(RAFT(args))
        # model.load_state_dict(torch.load(args.model))
        #
        # self.model = model.module
        # self.model.to(self.DEVICE)
        # self.model.eval()

        ## Load model
        checkpoint = torch.load(model_checkpoint_path)
        # checkpoint = torch.load("saved_check_point/pwcnet/IRR-PWC_sintel/checkpoint_latest.ckpt")
        state_dict = checkpoint['state_dict']
        state_dict_new = {}
        for key, value in state_dict.items():
            key = key.replace("_model.", "")
            state_dict_new[key] = value
        if model_name == 'irr-PWC_Net':
            self.model = irr_PWCNet(args=None)

        # ####
        # import networks.irr.models.pwcnet_occ_bi as pwc
        # #model = torch.nn.DataParallel(pwc.PWCNet(args=None))
        # model =pwc.PWCNet(args=None)
        # state_dict = checkpoint['state_dict']
        # state_dict_new = {}
        # for key, value in state_dict.items():
        #     key = key.replace("_model.", "")
        #     state_dict_new[key] = value
        #
        # model.load_state_dict(state_dict_new)
        ####

        self.model.load_state_dict(state_dict_new)
        self.model.cuda().eval()


    def __call__(self, inputs):
        img1 = inputs[0]
        img2 = inputs[1]
        im1_tensor = vision_transforms.transforms.ToTensor()(img1.astype(np.float32) / np.float32(255.0)).unsqueeze(0).cuda()
        im2_tensor = vision_transforms.transforms.ToTensor()(img2.astype(np.float32) / np.float32(255.0)).unsqueeze(0).cuda()
        with torch.no_grad():
            ## Forward pass
            input_dict = {'input1' : im1_tensor, 'input2' : im2_tensor}
            output_dict = self.model.forward(input_dict)
            flow = output_dict['flow'].squeeze(0)#.detach().cpu().numpy()
            flow_perm = flow.permute(1,0,2)#[0,:,:,:]
            flow_perm = flow_perm.permute(0, 2, 1).detach().cpu().numpy()

        return flow_perm
