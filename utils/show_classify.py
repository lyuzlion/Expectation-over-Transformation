import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import PIL
import numpy as np

import tempfile
from urllib.request import urlretrieve
import tarfile
import os

import json
import matplotlib.pyplot as plt

def show_classify(inception, img, imagenet_labels, correct_class=None, target_class=None):
     # 输入一张三通道图片，像素归一化，用模型预测，然后可视化它被预测成各种类的概率
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    fig.sca(ax1)
    
    tmp   = img.transpose((2,0,1))
    tmp_x = np.expand_dims(tmp, axis=0)
    
    p = np.squeeze( inception( torch.tensor(tmp_x)).detach().cpu().numpy() ) # sess.run(probs, feed_dict={image: img})[0]
    ax1.imshow(img)
    fig.sca(ax1)
    
    p = torch.sigmoid(torch.tensor(p)).cpu().numpy()
    # print("p: ", p)
    topk = list(p.argsort()[-10:][::-1])
    
    topprobs = p[topk]
    
    print(topprobs)
    
    barlist = ax2.bar(range(10), topprobs)
    if target_class in topk:
        barlist[topk.index(target_class)].set_color('r')
    if correct_class in topk:
        barlist[topk.index(correct_class)].set_color('g')
    plt.sca(ax2)
    plt.ylim([0, 1.1])
    plt.xticks(range(10),
               [imagenet_labels[i][:15] for i in topk],
               rotation='vertical')
    fig.subplots_adjust(bottom=0.2)
    plt.show()