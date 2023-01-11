from PIL import Image, ImageDraw
import os 

dest_path='/media/ssavian/Data1/EVALUATION/WACV_vis_fig1/inputs'
if not os.path.exists(dest_path):
    os.makedirs(dest_path)



im = Image.new('RGB', (1024, 1024), (51, 255, 255))
draw = ImageDraw.Draw(im)


x = 1024/2
y= 1024/2
r=120
leftUpPoint = (x-r, y-r)
rightDownPoint = (x+r, y+r)
twoPointList = [leftUpPoint, rightDownPoint]

draw.ellipse(twoPointList, fill=(255,0,0,255))

# draw.ellipse((100, 100, 150, 200), fill=(255, 0, 0), outline=(0, 0, 0))
# draw.rectangle((200, 100, 300, 200), fill=(0, 192, 192), outline=(255, 255, 255))
# draw.line((350, 200, 450, 100), fill=(255, 255, 0), width=10)

im.save(os.path.join(dest_path,'input_L.png'))

im = Image.new('RGB', (1024, 1024), (51, 255, 255))
draw = ImageDraw.Draw(im)

x = 1024 - 2*r
y= 1024/2
r=120
leftUpPoint = (x-r, y-r)
rightDownPoint = (x+r, y+r)
twoPointList = [leftUpPoint, rightDownPoint]

draw.ellipse(twoPointList, fill=(255,0,0,255))


im.save(os.path.join(dest_path,'input_R.png'))