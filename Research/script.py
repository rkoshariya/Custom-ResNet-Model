import timm
import cv2
import os
import torch
from torchvision.models.feature_extraction import create_feature_extractor
import random
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
torch.ops.load_library("op.pyd")

from prepare_input import prepare

from render import show_images

model = timm.create_model('resnet18', pretrained=True)
model.eval();

current_path = os.getcwd()

folder = os.path.join(current_path, 'Challenge_images')
imgs = []

    # loop us implemented to read the images one by one from the loop.
    # listdir return a list containing the names of the entries
    # in the directory.
for filename in os.listdir(folder):
    img = cv2.imread((os.path.join(folder, filename)))
    img = cv2.resize(img, (416, 416),
               interpolation = cv2.INTER_NEAREST)
    imgs.append(img)

inps = [prepare(img, model.default_cfg['mean'], model.default_cfg['std'])
        for img in imgs]

return_nodes = ['layer1', 'layer2', 'layer3', 'layer4']

feat_ext = create_feature_extractor(model, return_nodes=return_nodes)
tensor_result =torch.zeros(50,512)
for i in inps:
    with torch.no_grad():
        out = feat_ext(inps[i])
    
    m = nn.AdaptiveAvgPool2d((1,1))
    n = nn.AdaptiveAvgPool2d((1,1))
    o = nn.AdaptiveAvgPool2d((1,1))
    p = nn.AdaptiveAvgPool2d((1,1))

    layer1_out = m(out['layer1']).view(1,-1)
    layer2_out = n(out['layer2']).view(1,-1)
    layer3_out = m(out['layer3']).view(1,-1)
    layer4_out = m(out['layer4']).view(1,-1)

    x = torch.ops.custom_namespace.op(layer1_out,layer2_out,layer3_out,layer4_out)
    tensor_result[i] = x
    i = i + 1
image_list =[]
for i in range(0,50):
    count = 0
    for j in range(0,50):
        cos = torch.nn.CosineSimilarity(dim=0 , eps = 1e-6)
        output = cos(tensor_result[i],tensor_result[j])
        if(output>=0.75):
            count = count + 1
        else:
            pass
    if count <= 1:
        image_list.append(i)
    
print("Unique images are:")
for x in image_list:
    print(image_list[x])





