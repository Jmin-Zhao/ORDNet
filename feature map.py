import torch
from classic_models.mobilenet_v3 import mobilenet_v3_small
import torch.nn as nn
from torchvision import transforms

import matplotlib.pyplot as plt

def viz(module, input):
    x = input[0][0]
    #最多显示4张图
    min_num = np.minimum(4, x.size()[0])
    for i in range(min_num):
        plt.subplot(1, 4, i+1)
        plt.imshow(x[i].cpu())
    plt.show()


import cv2
import numpy as np
def main():

    t = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = cv2.imread('/home/zjm/programe/Deep-Learning-Classification/dataset/test/12-100%/0_44-1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = t(img).unsqueeze(0).to(device)
    model = mobilenet_v3_small(num_classes=2).to(device)
    for name1, m in model.named_children():
        # if not isinstance(m, torch.nn.ModuleList) and \
        #         not isinstance(m, torch.nn.Sequential) and \
        #         type(m) in torch.nn.__dict__.values():
        # 这里只对卷积层的feature map进行显示
        # if name=='features' and isinstance(m, torch.nn.Conv2d):
        #     m.register_forward_pre_hook(viz)
        # if name1, m =='avgpool':
            if name1 == 'avgpool' and isinstance(m, torch.nn.AdaptiveAvgPool2d):
                m.register_forward_pre_hook(viz)
            # for name2,n in m.named_children():
            #     if name2=='12':
            #         for name3, l in n.named_children():
            #             if name3[:3] == 'con' and isinstance(l, torch.nn.Conv2d):
            #                 l.register_forward_pre_hook(viz)
    with torch.no_grad():
        model(img)

if __name__ == '__main__':
    main()