#高斯模糊
# from PIL import Image
# import numpy as np
# from scipy.ndimage import filters
# import matplotlib.pyplot as plt
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
#
# im = Image.open('test.jpg')
# im_gray = im.convert('L')
# im_array = np.array(im_gray)
#
# index = 1
# plt.figure(figsize=(10, 3))
#
# plt.subplot(1, 4, index)
# plt.imshow(im_array, cmap='gray')
# plt.title('原始图像')
# plt.axis('off')
# index += 1
#
# for sigma in (2, 5, 10):
#     im_blur = filters.gaussian_filter(im_array, sigma)
#
#     plt.subplot(1, 4, index)
#     plt.imshow(im_blur, cmap='gray')
#     plt.title(f'σ={sigma}')
#     plt.axis('off')
#     index += 1
#
# plt.tight_layout()
# plt.show()
from numpy import histogram


#创建缩略图
# import os
# import glob
# from PIL import Image
#
# search_path = os.path.join('C:\\Users\\lin\\Desktop\\latexpic\\test', '*.jpg')
# images = glob.glob(search_path)
#
# for img_path in images:
#     img = Image.open(img_path)
#     img.thumbnail((80, 80))
#     print(img.format, img.size, img.mode)
#
#     thumbnail_filename = os.path.splitext(os.path.basename(img_path))[0] + "_1.jpg"
#     thumbnail_path = os.path.join('C:\\Users\\lin\\Desktop\\latexpic\\test', thumbnail_filename)
#     img.save(thumbnail_path, 'JPEG')

#轮廓线绘制
# from PIL import Image
# import numpy as np
# from scipy.ndimage import filters
# import matplotlib.pyplot as plt
# import cv2
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# im = Image.open('test.jpg')
# im_gray = im.convert('L')
# im_array = np.array(im_gray)
#
# plt.figure(figsize=(18, 6))
# sigmas = (2, 5, 10)
# index = 1
#
# for sigma in sigmas:
#     im_blur = filters.gaussian_filter(im_array, sigma)
#
#     _, im_binary = cv2.threshold(im_blur, 128, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(im_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contour_img = np.zeros_like(im_binary)
#     cv2.drawContours(contour_img, contours, -1, (255), 2)
#
#     plt.subplot(1, 3, index)
#     plt.imshow(contour_img, cmap='gray')
#     plt.title(f'σ={sigma}下的轮廓图')
#     plt.axis('off')
#     index += 1
#
# plt.tight_layout()
# plt.show()

# #直方图均衡化
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']
# 
# def histeq(im, nbr_bins=256):
#     imhist, bins = np.histogram(im.flatten(), nbr_bins, range=(0, 256))
#     cdf = imhist.cumsum()
#     cdf = cdf * 255.0 / cdf[-1]
#     im2 = np.interp(im.flatten(), bins[:-1], cdf)
#     im2 = im2.reshape(im.shape)
#     return im2,cdf
# 
# im = np.array(Image.open('test.jpg').convert('L'))
# im2, cdf_normalized = histeq(im)
# plt.figure(figsize=(10, 5))
# plt.subplot(121), plt.imshow(im, cmap='gray'), plt.title('原图')
# plt.subplot(122), plt.imshow(im2, cmap='gray'), plt.title('变换后')
# plt.show()

# #图像导数
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import filters
# plt.rcParams['font.sans-serif'] = ['SimHei']
# im = np.array(Image.open('test.jpg').convert('L'))
# fig, axs = plt.subplots(1, 4, figsize=(12, 3))
#
# axs[0].axis('off')
# axs[0].set_title(u'(a)原图')
# axs[0].imshow(im, cmap='gray')
#
# imx = filters.sobel(im, 1)  # 沿 x 轴应用 Sobel 滤波器
# imx_normalized = (imx - imx.min()) * 255 / (imx.max() - imx.min())
# axs[1].axis('off')
# axs[1].set_title(u'(b)x导数图像')
# axs[1].imshow(imx_normalized, cmap='gray')
#
# imy = filters.sobel(im, 0)  # 沿 y 轴应用 Sobel 滤波器
# imy_normalized = (imy - imy.min()) * 255 / (imy.max() - imy.min())
# axs[2].axis('off')
# axs[2].set_title(u'(c)y导数图像')
# axs[2].imshow(imy_normalized, cmap='gray')
#
# mag = np.sqrt(imx ** 2 + imy ** 2)
# mag_normalized = (mag - mag.min()) * 255 / (mag.max() - mag.min())
# axs[3].set_title(u'(d)梯度大小图像')
# axs[3].axis('off')
# axs[3].imshow(mag_normalized, cmap='gray')
#
# plt.show()

#复制粘贴、调整尺寸、旋转
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# plt.rcParams['font.sans-serif'] = ['SimHei']
# pil_im = Image.open('test.jpg')
# plt.subplot(221)
# plt.title(u'原图')
# plt.axis('off')
# plt.imshow(np.array(pil_im))
# pil_im = Image.open('test.jpg')
# box = (100, 100, 400, 400)
# region = pil_im.crop(box)
# region = region.transpose(Image.ROTATE_180)
# pil_im.paste(region, box)
# plt.subplot(222)
# plt.title(u'拷贝粘贴')
# plt.axis('off')
# plt.imshow(np.array(pil_im))
# size = (40, 30)
# pil_im = Image.open('test.jpg')
# pil_im = pil_im.resize(size)
# plt.subplot(223)
# plt.title(u'调整尺寸')
# plt.axis('off')
# plt.imshow(np.array(pil_im))
# pil_im = Image.open('test.jpg')
# pil_im = pil_im.rotate(45, expand=True)
# plt.subplot(224)
# plt.title(u'旋转45°')
# plt.axis('off')
# plt.imshow(np.array(pil_im))
# plt.show()

#基本绘图
# from PIL import Image
# from pylab import *
# plt.rcParams['font.sans-serif'] = ['SimHei']
# im = array(Image.open('test.jpg'))
# imshow(im)
# x = [1000,1000,4000,4000]
# y = [2000,500,2000,500]
# plot(x,y,'r*')
# plot(x[:2],y[:2])
# title('带有坐标轴的包含点和一条线段的图像')
# show()

#轮廓线与直方图
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
#
# im = np.array(Image.open('test.jpg').convert('L'))
# fig, axs = plt.subplots(1, 2, figsize=(10, 4))
#
# axs[0].contour(im, origin='image')
# axs[0].axis('off')
#
# axs[1].hist(im.flatten(), bins=128, color='gray')
#
# plt.tight_layout()
# plt.show()

