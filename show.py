import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import PIL
import numpy as np
from utils.show_classify import *
import tempfile
from urllib.request import urlretrieve
import tarfile
import os

import json
import matplotlib.pyplot as plt

inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1).eval()


img_path = './imgs/cat.jpg'
img = PIL.Image.open(img_path)
big_dim = max(img.width, img.height)
wide = img.width > img.height
new_w = 299 if not wide else int(img.width * 299 / img.height)
new_h = 299 if wide else int(img.height * 299 / img.width)
img = img.resize((new_w, new_h)).crop((0, 0, 299, 299))
img = (np.asarray(img) / 255.0).astype(np.float32) # 归一化为0-1之间的数

img_class = 281

plt.imshow(img)
plt.axis('off')
# plt.show()

# print(img.shape)
tmp = img.transpose((2,0,1))
# print(tmp.shape)

tmp_x = np.expand_dims(tmp, axis=0)

# print(torch.tensor(tmp_x).shape)
probs = inception(torch.tensor(tmp_x))
# print(probs.shape) # 共1000类

# print(torch.argmax(probs, axis=1)) # 输出281，分类正确

with open('./utils/ImageNetLabels.json', 'r') as f:
    imagenet_labels = json.load(f) # 以json格式存的标签名字

# print(imagenet_labels) # 一维数组，标签名字

show_classify(inception, img, imagenet_labels, correct_class=img_class)

