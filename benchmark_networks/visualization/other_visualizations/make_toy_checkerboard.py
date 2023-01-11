from PIL import Image, ImageDraw
import os 

dest_path='/media/ssavian/Data1/EVALUATION/WACV_vis_fig1/inputs'
if not os.path.exists(dest_path):
    os.makedirs(dest_path)

from PIL import Image
import numpy as np

n = 128 # size of one element, row = 8*n, chessboard = 8*n x 8*n

segment_black = np.zeros(shape = [n,n])
segment_white = np.ones(shape = [n,n])*255
chessboard = np.hstack((segment_black,segment_white))
for i in range(3):
    chessboard = np.hstack((chessboard,segment_black))
    chessboard = np.hstack((chessboard,segment_white))
temp = chessboard
for i in range(7):
    chessboard = np.concatenate((np.fliplr(chessboard),temp))
img = Image.fromarray(chessboard.astype(np.uint8))


img.save(os.path.join(dest_path,'checkerboard.png'))