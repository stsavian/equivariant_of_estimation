import inference_scripts.inference_RAFT as inference_RAFT
import generate_matrix_and_stats.matrix_from_file_path_and_model_instance as matrix_from_file_path_and_model_instance
import argparse
from imageio import imread,imwrite
import utils.flow
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
frL_pth='/media/ssavian/Data1/EVALUATION/WACV_vis_fig1/inputs/input_Lcheck.png'
frR_pth='/media/ssavian/Data1/EVALUATION/WACV_vis_fig1/inputs/input_Rcheck.png'
flo_pth='/home/ssavian/training/KITTI2015/training/flow_occ/000193_10.test'

frL_pth='/media/ssavian/Data/DATASETS/sintel/training/final/ambush_2/frame_0013.png'
frR_pth='/media/ssavian/Data/DATASETS/sintel/training/final/ambush_2/frame_0014.png'
flo_pth='/media/ssavian/Data/DATASETS/sintel/training/flow/ambush_2/frame_0014.flo'

dest_path='/media/ssavian/Data1/EVALUATION/WACV_vis_fig1_b/'
if not os.path.exists(dest_path):
    os.makedirs(dest_path)
train_things_pth='/media/ssavian/Data/BACKUP_v2/BACKUP_models/transfer/MODELS_things/raft_things_original_param/120000_raft_things_original_param.pth'
# train_things_pth='/home/ssavian/optical_flow_networks/RAFT/models/raft-things.pth'
# train_kitti_pth='/home/ssavian/optical_flow_networks/RAFT/models/raft-kitti.pth'

train_pth= train_things_pth
type='TEST'


args_raft_original = argparse.Namespace(alternate_corr=False, corr_levels=4, corr_radius=4, dropout=0, mixed_precision=False, \
                               model=[train_pth], path=None, small=False)
RAFT_inference_original = inference_RAFT.inference(args_raft_original)

out,out_star,GND = matrix_from_file_path_and_model_instance.generate_o_star(frL_pth, frR_pth, flo_pth, RAFT_inference_original,rotate_90_degrees=False)

##save everything
#destination

##get visualizations
frL = np.asarray(imread(frL_pth))  # certain frames contain opacity layer
frR = np.asarray(imread(frR_pth))
if flo_pth[-4:] == '.pfm':
    target = utils.flow.readPFM(flo_pth)[0][:, :, :2]
elif flo_pth[-4:] == '.png':
    # target =  Image.open(flow1_pth)
    # target = np.array(target).astype(np.float32)[:,:,:2]#check@!!!!!!!!
    target = utils.flow.read_png_flow(flo_pth)[0]
else:
    target = cv2.readOpticalFlow(flo_pth)

## SAVING AND VISUALIZING INPUTS
imwrite(os.path.join(dest_path,'frL.png'),frL)
imwrite(os.path.join(dest_path,'frR.png'),frR)
#imwrite(os.path.join(dest_path,type+'gnd_vis.png'),utils.flow.flow_to_png_middlebury(GND))

##SAVING AND VISUALIZING ESTIMATES
imwrite(os.path.join(dest_path,type+'out_vis.png'),utils.flow.flow_to_png_middlebury(out))
imwrite(os.path.join(dest_path,type+'out_star_vis.png'),utils.flow.flow_to_png_middlebury(out_star))

##SAVING THE ERROR MAP
err = out+out_star
err_mag =np.sqrt(np.square(err[:,:,0])+np.square(err[...,1]))
err_mag=(err_mag)
norm_mag= plt.Normalize(vmin=err_mag.min(), vmax=err_mag.max())
plt.imsave(os.path.join(dest_path,type+'error_map_mag.png'),np.multiply(norm_mag(err_mag.astype(np.float32)),255).astype(np.uint8),cmap='Reds')
#norm_mag= plt.Normalize(vmin=err_mag.min(), vmax=err_mag.max())
#plt.imsave(os.path.join(dest_path,'error_map_mag.png'),np.multiply((err_mag.astype(np.float32)),255).astype(np.uint8),cmap='bwr')


# err=np.dstack((err,np.zeros(err[:,:,0].shape)))
# norm = plt.Normalize(vmin=err.min(), vmax=err.max())
# plt.imsave(os.path.join(dest_path,'error_map2.png'),np.multiply(norm(err.astype(np.float32)),255).astype(np.uint8),cmap='gist_yarg')
# imm=Image.fromarray(np.multiply(norm(err.astype(np.float32)),255).astype(np.uint8), 'RGB')
# imwrite(os.path.join(dest_path,'error_map1.png'),imm,cmap='bwr')
#imwrite(os.path.join(dest_path,'error_map.png'),norm(err.astype(np.float32)),cmap='coolwarm')



print('dkjdjd')

