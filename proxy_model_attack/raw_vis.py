import rawpy
import imageio
import matplotlib.pylab as plt
import numpy as np
import os

# raw_dir = os.path.join('images')
# raw = np.asarray(imageio.imread(os.path.join(raw_dir, '18' + '.png')), dtype=np.double)
# raw.tofile('newdata.RAW')
#
raw = rawpy.imread('IMG_1114.CR2')
# #直接调用postprocess可能出现偏色问题
# #rgb = raw.postprocess()
#
# #以下两行可能解决偏色问题，output_bps=16表示输出是 16 bit (2^16=65536)需要转换一次
im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
print(im.shape)
rgb = np.float32(im / 65535.0 * 255.0)
rgb = np.asarray(rgb,np.uint8)
#
imageio.imsave('image.jpg', rgb)
