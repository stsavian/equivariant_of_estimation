#import inference
import cv2
import numpy as np
from imageio import imread,imsave
import utils.flow
#import imbalance_csv_plots.imbalance_utils_v1 as imbalance_utils
#import imbalance_csv_plots.utils_OF as utils_OF
#import torch
from PIL import Image

def generate_pred(fr1,fr2,model_instance):
    inputs = [fr1.copy(), fr2.copy()]
    out_flo = model_instance(inputs)
    return out_flo
def generate_o_star(frame1_pth,frame2_pth,flow1_pth,inf,rotate_90_degrees=False,test_mean=False,KITTI_only_valid=False):#remember

    # reading the files
    if flow1_pth[-4:]=='.pfm':
        target = utils.flow.readPFM(flow1_pth)[0][:,:,:2]
    elif flow1_pth[-4:]=='.png':
        # target =  Image.open(flow1_pth)
        # target = np.array(target).astype(np.float32)[:,:,:2]#check@!!!!!!!!
        target = utils.flow.read_png_flow(flow1_pth)[0]# zero contain the OF #one contain the valid index
        valid = utils.flow.read_png_flow(flow1_pth)[1]


    else:
        target = cv2.readOpticalFlow(flow1_pth)

    fr1 = np.asarray(imread(frame1_pth))#certain frames contain opacity layer
    fr2 = np.asarray(imread(frame2_pth))


    if len(fr1.shape)==2:
        fr1 = np.dstack([fr1] * 3)
        #print('grayscale')
    if len(fr2.shape)==2:
        fr2 = np.dstack([fr2] * 3)

    if fr1.shape[2]==4:
        fr1=fr1[...,:3]
    if fr2.shape[2] == 4:
        fr2 = fr2[..., :3]

    if rotate_90_degrees == True:
        fr1 = np.rot90(fr1.copy(),1)
        fr2 = np.rot90(fr2.copy(), 1)
        target = np.rot90(target.copy(), 1)
        target = np.stack((target[:,:,1],-target[:,:,0]),axis=2)

    out = generate_pred(fr1, fr2, inf)

    fr1_180 = np.rot90(fr1.copy(), 2)
    fr2_180 = np.rot90(fr2.copy(), 2)

    out_180 = generate_pred(fr1_180, fr2_180, inf)
    Tout_180 = np.rot90(out_180.copy(), 2)
    #print('here')

    if KITTI_only_valid or flow1_pth[-4:]=='.png':
        # valid_areas=target[np.concatenate((valid.copy(), valid.copy()), axis=2) == True]
        # out=out[valid_areas]
        # Tout_180=Tout_180[valid_areas]
        # target=target[valid_areas]
        debug=False
        if debug:
            import utils.debug as debug
            import matplotlib.pyplot as plt
            debug.save_image(target, 'target.png', 'OF')
            plt.imsave('/home/ssavian/training/plots_ironspeed/temp/valid.png', np.multiply((valid[..., 0].astype(np.float32)), 255).astype(np.uint8), cmap='bwr')
        invalid_areas=np.concatenate((valid.copy(), valid[:375, :1242, :].copy()), axis=2) == 0
        ddflow=False
        if ddflow:
            out=out[:375,:1242, :]
            Tout_180 = Tout_180[:375, :1242, :]
            target=target[:375, :1242, :]
        out[invalid_areas]=np.NaN
        Tout_180[invalid_areas]=np.NaN
        target[invalid_areas]=np.NaN
        if debug:
            debug.save_image(out, 'out.png', 'OF')
            debug.save_image(Tout_180, 'outstar.png', 'OF')
            debug.save_image(target, 'target_valid.png', 'OF')

    if test_mean:
        Tout_180_mean = np.mean( np.array([ -out.copy(), Tout_180.copy() ]), axis=0 )
        out_mean = np.mean( np.array([ out.copy(),-Tout_180.copy() ]), axis=0 )
        #print(out_mean.mean(), Tout_180_mean.mean())
    #return out,Tout_180, target
        return  out_mean, Tout_180_mean, target


    return out, Tout_180, target

def generate_O_Tud_Tlr_T180(frame1_pth,frame2_pth,flow1_pth,inf,rotate_90_degrees=False):

    # reading the files
    target = cv2.readOpticalFlow(flow1_pth)

    fr1 = np.asarray(imread(frame1_pth))[...,:3]#certain frames contain opacity layer
    fr2 = np.asarray(imread(frame2_pth))[...,:3]#certain frames contain opacity layer

    if len(fr1.shape)==2:
        fr1 = np.dstack([fr1] * 3)
        #print('grayscale')
    if len(fr2.shape)==2:
        fr2 = np.dstack([fr2] * 3)

    if rotate_90_degrees == True:
        fr1 = np.rot90(fr1.copy(),1)
        fr2 = np.rot90(fr2.copy(), 1)
        target = np.rot90(target.copy(), 1)
        target = np.stack((target[:,:,1],-target[:,:,0]),axis=2)


    out = generate_pred(fr1, fr2, inf)


    fr1_lr = np.fliplr(fr1.copy())
    fr2_lr = np.fliplr(fr2.copy())

    out_lr = generate_pred(fr1_lr, fr2_lr, inf)
    Tout_lr = np.fliplr(out_lr.copy())

    fr1_ud = np.flipud(fr1.copy())
    fr2_ud = np.flipud(fr2.copy())

    out_ud = generate_pred(fr1_ud, fr2_ud, inf)
    Tout_ud = np.flipud(out_ud.copy())

    fr1_180 = np.rot90(fr1.copy(), 2)
    fr2_180 = np.rot90(fr2.copy(), 2)

    out_180 = generate_pred(fr1_180, fr2_180, inf)
    Tout_180 = np.rot90(out_180.copy(), 2)

    return  out, Tout_lr, Tout_ud, Tout_180