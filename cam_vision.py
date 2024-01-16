
import json
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from gradcam import GradCAM, show_cam_on_image, center_crop_img
import classic_models
from classic_models.mobilenet_v3 import mobilenet_v3_small
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def main():
    # ----载入自己的模型，按照自己训练集训练的
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # gpu
    # model = classic_models.find_model_using_name('mobilenet_v3', 2).to(device)
    model = mobilenet_v3_small(num_classes=2)

    # load model weights
    model_weight_path = "/home/zjm/programe/Deep-Learning-Classification/results/weights/-Origin500-290.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))  # 加载权重
    model.eval()
    target_layers = [model.features[-1]]  # 拿到最后一个层结构

    # 载入模型，模型权重---载入的模型是按照ImageNet训练的（不是自己的模型是从torchvision中导入的）
    # model = models.mobilenet_v3_large(pretrained=True)
    # target_layers = [model.features[-1]]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    path = "/home/zjm/programe/Deep-Learning-Classification/cam"
    save_path = "/home/zjm/programe/Deep-Learning-Classification/test_result"
    for class_dir in os.listdir(path):
        class_path = path + '/' + class_dir
        scpath = save_path + '/' + class_dir
        assert os.path.exists(scpath), "file: '{}' dose not exist.".format(scpath)
        # lenth = len(os.listdir(class_dir))
        for img_path in os.listdir(class_path):
            spath = scpath + '/' + img_path
            img_path = class_path + '/' + img_path
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path).convert('RGB')
            img = np.array(img, dtype=np.uint8)
            # img = center_crop_img(img, 224)

            # [C, H, W]
            img_tensor = data_transform(img)
            # expand batch dimension
            # [C, H, W] -> [N, C, H, W]
            input_tensor = torch.unsqueeze(img_tensor, dim=0)
            # 实例化，输出模型，要计算的层
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
            # 感兴趣的label
            if class_dir == '0-12%':
                target_category = 0
            else:
                target_category = 1  # 0表示0-12%，1表示12-100%
            # target_category = 254  # pug, pug-dog
            # 计算cam图
            grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)  # 实例化
            # 将只传入的一张图的cam图提取出来
            grayscale_cam = grayscale_cam[0, :]
            # 变成彩色热力图的形式
            visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,  # 将原图缩放到[0,1]之间
                                              grayscale_cam,
                                              use_rgb=True)
            # 展示出来
            json_path = '/home/zjm/programe/Deep-Learning-Classification/dataset/class_indices.json'
            assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
            with open(json_path, "r") as f:
                class_indict = json.load(f)
            if target_category == 1:
                plt.title('Undried')
            else:
                plt.title('Dried')

            plt.imshow(visualization)
            # plt.savefig(spath, bbox_inches='tight')
            plt.show()


if __name__ == '__main__':
    main()

