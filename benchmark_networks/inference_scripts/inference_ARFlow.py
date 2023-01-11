import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from easydict import EasyDict
from torchvision import transforms
from networks.ARFlow.transforms import sep_transforms

from networks.ARFlow.utils.flow_utils import flow_to_image, resize_flow
from networks.ARFlow.utils.torch_utils import restore_model
from networks.ARFlow.models.pwclite import PWCLite
from utils.padder import InputPadder
#Namespace(img_list=['examples/img1.png', 'examples/img2.png'], model='checkpoints/KITTI15/pwclite_ar.tar', test_shape=[384, 640])
class Inference():
    def __init__(self, cfg):
        # self.cfg = cfg
        self.cfg = EasyDict(cfg)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu")
        self.model = self.init_model()
        self.input_transform = transforms.Compose([
            sep_transforms.Zoom(*self.cfg.test_shape),
            sep_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ])

    def init_model(self):
        model = PWCLite(self.cfg.model)
        # print('Number fo parameters: {}'.format(model.num_parameters()))
        model = model.to(self.device)
        model = restore_model(model, self.cfg.pretrained_model)
        model.eval()
        return model

    def __call__(self, imgs):
        #padder = InputPadder(imgs[0].shape)
        imgs = [self.input_transform(img).unsqueeze(0) for img in imgs]
        #image1, image2 = padder.pad(imgs[0], imgs[1])
        #imgs = [image1,image2]
        img_pair = torch.cat(imgs, 1).to(self.device)
        output = self.model(img_pair)
        #output = padder.unpad(output)
        out = output['flows_fw'][0].squeeze(0)
        #out1 = out.permute([2, 0, 1])
        out2 = out.permute([1, 2, 0])
        out3 = out2[6:-6,:,:]
        return out3.detach().cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='checkpoints/KITTI15/pwclite_ar.tar')
    parser.add_argument('-s', '--test_shape', default=[384, 640], type=int, nargs=2)
    parser.add_argument('-i', '--img_list', nargs='+',
                        default=['examples/img1.png', 'examples/img2.png'])
    args = parser.parse_args()

    cfg = {
        'model': {
            'upsample': True,
            'n_frames': len(args.img_list),
            'reduce_dense': True
        },
        'pretrained_model': args.model,
        'test_shape': args.test_shape,
    }

    ts = TestHelper(cfg)

    imgs = [imageio.imread(img).astype(np.float32) for img in args.img_list]
    h, w = imgs[0].shape[:2]

    flow_12 = ts.run(imgs)['flows_fw'][0]

    flow_12 = resize_flow(flow_12, (h, w))
    np_flow_12 = flow_12[0].detach().cpu().numpy().transpose([1, 2, 0])

    vis_flow = flow_to_image(np_flow_12)

    fig = plt.figure()
    plt.imshow(vis_flow)
    plt.show()
