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
base_pth='/home/ssavian/training/visualization_1_samples/data'
frL_pth=os.path.join(base_pth,'pair1_L.png')
frR_pth=os.path.join(base_pth,'pair1_R.png')
flo_pth=os.path.join(base_pth,'pair1_.flo')

# dest_path='/home/ssavian/training/plots_ironspeed/visualization/kaleidoscope_1'
# if not os.path.exists(dest_path):
#     os.makedirs(dest_path)


#train_chairs_pth='/home/ssavian/training/RAFT_trained_models/first_run/raft-chairs.pth'
train_things_pth='/home/ssavian/optical_flow_networks/RAFT/models/raft-things.pth'
train_FWDG_b1_things='/home/ssavian/training/transfer/things_mir_b10_fine_tune_raft-chairs_mir_10_FWDG_averaged_loss.pth'
# train_kitti_pth='/home/ssavian/optical_flow_networks/RAFT/models/raft-kitti.pth'

train_pth= train_FWDG_b1_things



args_raft_original = argparse.Namespace(alternate_corr=False, corr_levels=4, corr_radius=4, dropout=0, mixed_precision=False, \
                               model=train_pth, path=None, small=False)
RAFT_inference_original = inference_RAFT.inference(args_raft_original)

#out,out_star,GND = matrix_from_file_path_and_model_instance.generate_o_star(frL_pth, frR_pth, flo_pth, RAFT_inference_original,rotate_90_degrees=False)

dest_path='/home/ssavian/training/visualization/raft-things_FWDGb1_sintel/'

##save everything
#destination

def save_sintel_visualizations(model_inference,dataset_pth,dest_path,mode='final'):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    #path for clean or final
    if mode =='clean':
        frames_pth = 'training/clean/'
    elif mode=='final':
        frames_pth = 'training/final/'
    flow_pth   = 'training/flow/'

    frames_pth = os.path.join(dataset_pth,frames_pth)
    flow_pth = os.path.join(dataset_pth, flow_pth)
    fr_seq  = sorted(os.listdir(frames_pth))
    flo_seq = sorted(os.listdir(flow_pth))

    n_frames = 0
    for item in fr_seq:
        fr_names = sorted(os.listdir(os.path.join(frames_pth, item)))
        flo_names = sorted(os.listdir(os.path.join(flow_pth, item)))
        n_frames += len(fr_names)
        print(item, len(fr_names))

    print('TOT ' + str(n_frames) + 'frames')
    #all_rows = []
    list_all = []
    list_full_frame = []
    list_masked = []
    list_tud_tlr_t180 = []

    for item in fr_seq:
        print(item)
        fr_names = sorted(os.listdir(os.path.join(frames_pth, item)))
        flo_names = sorted(os.listdir(os.path.join(flow_pth, item)))
        for i in range(0,len(fr_names) - 1):  # starts from 1 ends at -1
            # print(i)
            row = {}
            fr_L = fr_names[i]
            fr_R = fr_names[i + 1]
            flo_i = flo_names[i]
            frL_pth = os.path.join(frames_pth, item, fr_L)
            frR_pth = os.path.join(frames_pth, item, fr_R)
            flo_pth = os.path.join(flow_pth, item, flo_i)

            out,out_star,GND = matrix_from_file_path_and_model_instance.generate_o_star(frL_pth, frR_pth, flo_pth, model_inference,rotate_90_degrees=False)
            print(flo_pth.split("/")[-2]+'_'+flo_pth.split("/")[-1][:-4])
            row_dest_path=os.path.join(dest_path, flo_pth.split("/")[-2]+'_'+flo_pth.split("/")[-1][:-4])#check
            frL, frR, target= get_input_matrix_and_G(frL_pth, frR_pth, flo_pth)
            save_inputs_and_G(frL, frR, GND, row_dest_path)
            save_O_and_Ostar_e_map(out,out_star,GND,row_dest_path)
    return
def get_input_matrix_and_G(frL_pth,frR_pth,flo_pth):
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
    return frL,frR,target
## SAVING AND VISUALIZING INPUTS
def save_inputs_and_G(frL,frR,GND,dest_path):
    imwrite(os.path.join(dest_path+'_frL.png'),frL)
    imwrite(os.path.join(dest_path+'_frR.png'),frR)
    imwrite(os.path.join(dest_path+'_G_vis.png'),utils.flow.flow_to_png_middlebury(GND))
    return
#def save output OF o,ostar,difference (O,Ostar,dest_pth,dest_names)
##get visualizations
def save_O_and_Ostar_e_map(out,out_star,GND,dest_path):
    ##SAVING AND VISUALIZING ESTIMATES
    imwrite(os.path.join(dest_path+'out_vis.png'), utils.flow.flow_to_png_middlebury(out))
    imwrite(os.path.join(dest_path+  'out_star_vis.png'), utils.flow.flow_to_png_middlebury(-out_star))
    ##SAVING THE ERROR MAP
    err = out + out_star
    err_mag = np.sqrt(np.square(err[:, :, 0]) + np.square(err[..., 1]))
    err_mag = (err_mag)
    norm_mag = plt.Normalize(vmin=err_mag.min(), vmax=err_mag.max())
    plt.imsave(os.path.join(dest_path + 'error_map_mag.png'),
    np.multiply(norm_mag(err_mag.astype(np.float32)), 255).astype(np.uint8), cmap='Reds')
    return



save_sintel_visualizations(RAFT_inference_original,'/home/ssavian/training/sintel/',dest_path,mode='final')


#norm_mag= plt.Normalize(vmin=err_mag.min(), vmax=err_mag.max())
#plt.imsave(os.path.join(dest_path,'error_map_mag.png'),np.multiply((err_mag.astype(np.float32)),255).astype(np.uint8),cmap='bwr')


# err=np.dstack((err,np.zeros(err[:,:,0].shape)))
# norm = plt.Normalize(vmin=err.min(), vmax=err.max())
# plt.imsave(os.path.join(dest_path,'error_map2.png'),np.multiply(norm(err.astype(np.float32)),255).astype(np.uint8),cmap='gist_yarg')
# imm=Image.fromarray(np.multiply(norm(err.astype(np.float32)),255).astype(np.uint8), 'RGB')
# imwrite(os.path.join(dest_path,'error_map1.png'),imm,cmap='bwr')
#imwrite(os.path.join(dest_path,'error_map.png'),norm(err.astype(np.float32)),cmap='coolwarm')




