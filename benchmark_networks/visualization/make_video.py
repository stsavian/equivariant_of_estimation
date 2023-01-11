import mediapy as media
import os
from PIL import Image
import argparse
import numpy as np

def absolute_file_paths(directory):
    path = os.path.abspath(directory)
    return [entry.path for entry in os.scandir(path) if entry.is_file()]
def make_video(args):
    new_file = args.video_folder[0]
    data = absolute_file_paths(args.input_folder[0])
    h,w,ch=np.asarray(Image.open(data[0])).shape

    
    with media.VideoWriter(
    new_file, shape=(h,w), fps=10) as writer:
        for i in range(0, len(data)):
            #flow_color = flow_vis.flow_to_color(data['arr_0'][i + s][:, :, 2:], convert_to_bgr=False)
            writer.add_image(np.asarray(Image.open(data[i]))[...,:3])
    media.show_video(media.read_video(new_file), height=90)
    return




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_folder', type=str, nargs='+')
    parser.add_argument('--input_folder', type=str, nargs='+')
    

    # parser.add_argument('--train_pth', type=str, nargs='+')
    # parser.add_argument('--mode', type=str, nargs='+')
    args = parser.parse_args()
    make_video(args)