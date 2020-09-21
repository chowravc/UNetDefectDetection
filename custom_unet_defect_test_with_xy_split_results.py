import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pims
import pathlib
import torch.optim as optim
from torch.autograd import Variable
import skimage as skm
import glob
from PIL import Image
print("Loaded packages.")

def double_conv(in_c, out_c):

	conv = nn.Sequential(
		nn.Conv2d(in_c, out_c, kernel_size = 3),
		nn.ReLU(inplace = True),
		nn.Conv2d(out_c, out_c, kernel_size = 3),
		nn.ReLU(inplace = True)
	)

	return conv

def crop_img(tensor, target_tensor):

	target_size = target_tensor.size()[2]
	tensor_size = tensor.size()[2]

	delta = tensor_size - target_size
	delta = delta // 2

	return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]

class UNetOld(nn.Module):

	def __init__(self):

		super(UNetOld, self).__init__()

		self.max_pool_2x2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

		self.down_conv_1 = double_conv(1, 64) # Only 1 channel at the moment
		self.down_conv_2 = double_conv(64, 128)
		self.down_conv_3 = double_conv(128, 256)
	
		self.up_trans_3 = nn.ConvTranspose2d(
			in_channels = 256,
			out_channels = 128,
			kernel_size = 2,
			stride = 2
		)

		self.up_conv_3 = double_conv(256, 128)

		self.up_trans_4 = nn.ConvTranspose2d(
			in_channels = 128,
			out_channels = 64,
			kernel_size = 2,
			stride = 2
		)

		self.up_conv_4 = double_conv(128, 64)

		self.out = nn.Conv2d(
			in_channels = 64,
			out_channels = 1,
			kernel_size = 1
		)

	def forward(self, image):

		# bs, c, h, w
		# Encoder
		x1 = self.down_conv_1(image)#
		x3 = self.max_pool_2x2(x1)

		x3 = self.down_conv_2(x3)#
		x5 = self.max_pool_2x2(x3)

		x5 = self.down_conv_3(x5)#

		# Decoder

		x = self.up_trans_3(x5)
		y = crop_img(x3, x)
		x = self.up_conv_3(torch.cat([x, y], 1))

		x = self.up_trans_4(x)
		y = crop_img(x1, x)
		x = self.up_conv_4(torch.cat([x, y], 1))
	
		x1, x3, x5, y = None, None, None, None,

		x = self.out(x)
		return x


def main():

	"""### Choose Model to Load"""
	PATH = './DefectUNetWeightsStorage' + '/unetTest19thSept_1.pth'
	device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
	print("Model running on device " +str(device))

	uModel = UNetOld()
	uModel.to(device)
	uModel.load_state_dict(torch.load(PATH))
	print("Loaded Model.")

	"""### Testing Images"""
	print("Enter path of image to run model on.")
	imagePath = input("(Return empty to run default r3_1000.tif): ")
	if imagePath == "":
		imagePath = "./DefectUNetWeightsStorage/testImages/" + "r3_1000.tif"

	# Apply sigmoid layer to output?
	sig = True

	# RGB image?
	rgbFlag = False

	imsize = 2*(84+2)

	loader = transforms.Compose([transforms.Resize(imsize), transforms.CenterCrop((imsize,imsize)), transforms.ToTensor()])

	def cropTen(a, b): #a<=b
	  sa1, sa2 = len(a), len(a[0])
	  sb1, sb2 = len(b), len(b[0])
	  d1, d2 = (sb1-sa1)//2, (sb2-sa2)//2
	  return b[d1:sb1-d1, d2:sb2-d2]

	def image_loader(image_name):
		"""load image, returns cuda tensor"""
		image = Image.open(image_name)
		image = loader(image).float()
		image = Variable(image, requires_grad=True)
		image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
		return image.to(device)  #assumes that you're using GPU

	def grey2rgb_image_loader(image_name):
		"""load image, returns cuda tensor"""
		image = Image.open(image_name)
		image = loader(image).float()
		image = Variable(image, requires_grad=True)
		image = torch.stack([image[0], image[0], image[0]])
		image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
		return image.to(device)  #assumes that you're using GPU

	def rgb2grey_image_loader(image_name):
		"""load image, returns cuda tensor"""
		image = Image.open(image_name)
		image = loader(image).float()
		image = Variable(image, requires_grad=True)
		image = (image[0] + image[1] + image[2])/3
		image = image.unsqueeze(0).unsqueeze(0)  #this is for VGG, may not be needed for ResNet
		return image.to(device)  #assumes that you're using GPU

	if not rgbFlag:
	  testImage = image_loader(imagePath)
	else:
	  testImage = rgb2grey_image_loader(imagePath)

	print("Loaded image.")

	if not sig:
	  testOutput = uModel(testImage)[0][0].to("cpu").detach().numpy()
	else:
	  testOutput = torch.sigmoid(5*uModel(testImage))[0][0].to("cpu").detach().numpy()

	fig1, ax1 = plt.subplots()

	testImage = cropTen(testOutput, testImage[0][0].to("cpu").detach().numpy())
	masked = np.ma.masked_where(testOutput == 1, testOutput)

	a = ax1.imshow(testImage, 'gray')

	fig2, ax2 = plt.subplots()
	ax2.imshow(testImage)
	b = ax2.imshow(testOutput, interpolation='none', alpha=0.4, cmap='gray')
	plt.colorbar(b)

	fig3, ax3 = plt.subplots()
	c = ax3.imshow(testOutput)
	plt.colorbar(c)

	fig1.show()
	fig2.show()
	fig3.show()

	save = input("Save figures? (yes/no) ")
	if save == "yes":
		fig2.savefig('outputOverlay.png', dpi=300)
		fig3.savefig('outputMask.png', dpi=300)
		print("Output saved.")

if __name__ == '__main__':
	main()