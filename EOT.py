import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import PIL
import numpy as np
from utils.show_classify import *
import json
import matplotlib.pyplot as plt

with open('./utils/ImageNetLabels.json', 'r') as f:
    imagenet_labels = json.load(f) # 以json格式存的标签名字
img_class = 281
img_path = './imgs/cat.jpg'
img = PIL.Image.open(img_path)
big_dim = max(img.width, img.height)
wide = img.width > img.height
new_w = 299 if not wide else int(img.width * 299 / img.height)
new_h = 299 if wide else int(img.height * 299 / img.width)
img = img.resize((new_w, new_h)).crop((0, 0, 299, 299))
img = (np.asarray(img) / 255.0).astype(np.float32) # 归一化为0-1之间的数


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1).to(device)
model.eval()

demo_epsilon = 2.0/255.0 # a really small perturbation
demo_lr = 1e-1
demo_steps = 15
demo_target = [924] # "guacamole"

alpha = 2/255
epsilon = 8/255

X = torch.tensor(np.expand_dims(img.transpose((2,0,1)), axis=0)).to(device) # (3, xxx, xxx)
X_adv = X.clone().detach().to(device)
criterion = nn.CrossEntropyLoss().to(device)


for i in range(demo_steps):
    X_adv.requires_grad = True
    average_loss = 0
    for j in range(20):
        pred = model(T.functional.rotate(X_adv, angle=(torch.rand(1) * 40 - 20).item())) # torch.Size([1, 1000])
        loss = criterion(pred, torch.tensor(demo_target).to(device))
        average_loss += loss / 10

    average_loss.backward() 
    X_adv.data = X_adv - alpha * torch.sign(X_adv.grad)
    X_adv.data = torch.min(X_adv, X+epsilon)
    X_adv.data = torch.max(X_adv, X-epsilon)
    X_adv.data = torch.clamp(X_adv, min=0, max=1)
    if i % 4 == 0: 
        print( f'[Iteration {i}] Loss: {loss.item()}' )

x_adv_star = X_adv[0]

plt.imshow(x_adv_star.cpu().detach().numpy().transpose(1, 2, 0))
plt.show()
rotated = T.functional.rotate(x_adv_star, angle=(torch.rand(1) * 40 - 20).item())


plt.imshow(rotated.cpu().detach().numpy().transpose(1, 2, 0))
plt.show()

pix_diff = np.abs(X_adv[0].cpu().detach().numpy() - X[0].cpu().detach().numpy()).transpose(1, 2, 0)
pix_diff *= 1 / np.max(pix_diff)
plt.imshow(pix_diff); 
plt.colorbar()
plt.show()

print(rotated.shape)
show_classify(model.cpu(), rotated.cpu().detach().numpy().transpose(1, 2, 0), imagenet_labels, correct_class=img_class, target_class=demo_target)


