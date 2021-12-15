import torch

from torchvision import transforms
from torchvision.utils import save_image

import matplotlib.pyplot as plt

from PIL import Image

import cv2

import numpy as np

import os



img_list = os.listdir("RawTest")

diff_summs, diff_means = [] , []

for i in img_list:
    img_path_raw = "RawTest/"+i
    img = Image.open(img_path_raw)

    convert_tensor=transforms.ToTensor()



    tensor_img=convert_tensor(img)
    tensor_img_resize=transforms.functional.resize(tensor_img,size=[64])



    print(tensor_img_resize.shape)
    print(tensor_img_resize.mean())
    print(tensor_img_resize.sum())
    print(tensor_img_resize.max())
    print(tensor_img_resize.min())
    #print(tensor_img_resize)
    #plt.imshow(tensor_img_resize.permute(1,2,0))

    #plt.show()

    raw = cv2.imread(img_path_raw)



    transform = transforms.Compose([transforms.ToTensor()])





    img_raw = cv2.resize(raw, (64, 64), interpolation = cv2.INTER_AREA)
    rawImage = transform(img_raw)

    print(rawImage.shape)
    print(rawImage.mean())
    print(rawImage.sum())
    print(rawImage.max())
    print(rawImage.min())

    rdner = np.transpose(rawImage.numpy(), (1, 2, 0))
    #plt.imshow(cv2.cvtColor(rdner, cv2.COLOR_BGR2RGB))

    #plt.show()

    diff_sum=tensor_img_resize.sum()-rawImage.sum()
    diff_mean=tensor_img_resize.mean()-rawImage.mean()
    diff_summs.append(diff_sum)
    diff_means.append(diff_mean)

    save_image(tensor_img,"FinalImage/"+i)

    break

print(np.mean(diff_summs))
print(np.mean(diff_means))