from PIL import Image, ImageDraw
import os 
import numpy as np

def make_checkerboard(n):
    #n = 128 # size of one element, row = 8*n, chessboard = 8*n x 8*n

    segment_black = np.zeros(shape = [n,n,3])
    segment_white = np.ones(shape = [n,n,3])*255
    chessboard = np.hstack((segment_black,segment_white))
    for i in range(3):
        chessboard = np.hstack((chessboard,segment_black))
        chessboard = np.hstack((chessboard,segment_white))
    temp = chessboard
    for i in range(7):
        chessboard = np.concatenate((np.fliplr(chessboard),temp))
    img = Image.fromarray(chessboard.astype(np.uint8))
    return img

dest_path='/media/ssavian/Data1/EVALUATION/WACV_vis_fig1/inputs_left'
if not os.path.exists(dest_path):
    os.makedirs(dest_path)



#im = Image.new('RGB', (1024, 1024), (51, 255, 255))
im = make_checkerboard(128)
draw = ImageDraw.Draw(im)


x = 1024/2
y= 1024/2
r=128
leftUpPoint = (x-r, y-r)
rightDownPoint = (x+r, y+r)
twoPointList_imgL = [leftUpPoint, rightDownPoint]

draw.ellipse(twoPointList_imgL, fill=(255,0,0,255))



# draw.ellipse((100, 100, 150, 200), fill=(255, 0, 0), outline=(0, 0, 0))
# draw.rectangle((200, 100, 300, 200), fill=(0, 192, 192), outline=(255, 255, 255))
# draw.line((350, 200, 450, 100), fill=(255, 255, 0), width=10)


im.save(os.path.join(dest_path,'input_Lcheck.png'))

#im = Image.new('RGB', (1024, 1024), (51, 255, 255))
im = make_checkerboard(128)
draw = ImageDraw.Draw(im)

new_x = 1024 - 2*r
new_x =  + 2*r

y= 1024/2
r=128
leftUpPoint = (new_x-r, y-r)
rightDownPoint = (new_x+r, y+r)
twoPointList = [leftUpPoint, rightDownPoint]

draw.ellipse(twoPointList, fill=(255,0,0,255))

im.save(os.path.join(dest_path,'input_Rcheck.png'))
####################################################
###################OPTICAL FLOW####################
###################################################
h,w=im.size
OF = Image.new('F', im.size, (0))
draw_of = ImageDraw.Draw(OF)
draw_of.ellipse(twoPointList_imgL, fill=(new_x-x))

OF_out=np.stack((np.asarray(OF),np.zeros(np.asarray(OF).shape)),axis=2)
import utils.flow as flow_utils
flow_utils.write_flow(os.path.join(dest_path,'gnd.flo'),OF_out)


