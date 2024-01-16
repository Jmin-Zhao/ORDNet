import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from sklearn.metrics import confusion_matrix, accuracy_score

from classic_models.mobilenet_v3 import mobilenet_v3_small
from classic_models.shufflenet_v2 import shufflenet_v2_x0_5
from classic_models.resnet import resnet50
from classic_models.resnet import resnet101
from utils.train_engin import confusionMatrix, p_and_r


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [
            # transforms.Resize(640),
            # transforms.Resize([640, 640]),
            # transforms.CenterCrop(640),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
         #    transforms.Resize((2560, 2560)),
         # transforms.CenterCrop(1280),
         # transforms.ToTensor(),
         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    # img_path = '/home/zjm/python/Deep-Learning-Classification-Models-Based-CNN-or-Attention/data_b/test/b/0-20.96.jpg'
    # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # img = Image.open(img_path)
    #
    # plt.imshow(img)
    # # [N, C, H, W]
    # img = data_transform(img)
    # # expand batch dimension
    # img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = '/home/zjm/programe/Deep-Learning-Classification/dataset/class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = mobilenet_v3_small(num_classes=2).to(device)
    # model = shufflenet_v2_x0_5(num_classes=2).to(device)
    # model = resnet50(num_classes=2).to(device)
    # model = efficientnetv2_s(num_classes=2).to(device)

    # load model weights
    weights_path = "/home/zjm/programe/Deep-Learning-Classification/results/weights/mobilenet_v3/muticolor500-140.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))
    file_path = "/home/zjm/programe/Deep-Learning-Classification/dataset/test/"
    # load image

    t_labels = []
    p_labels = []
    for class_dir in os.listdir(file_path):
        classsort = class_dir
        count = 0
        class_dir = file_path + '/' +class_dir
        # lenth = len(os.listdir(class_dir))
        for img_path in os.listdir(class_dir):
            img_path = class_dir + '/' +img_path
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path)

            plt.imshow(img)
            # [N, C, H, W]
            img = data_transform(img)
            # imge = transforms.ToPILImage()(img)
            # plt.imshow(imge)
            # plt.show()
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)
            model.eval()

            with torch.no_grad():
                # predict class
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()


            # if class_indict[str(predict_cla)] == classsort :
            #     count = count + 1
            if classsort[0] == '5':
                t_labels.extend('2')
            else:
                t_labels.extend(classsort[0])
            # p_labels.extend(list((class_indict[str(predict_cla)])[0])[0])
            if class_indict[str(predict_cla)][0] == '5':
                p_labels.extend('2')
            else:
                p_labels.extend(class_indict[str(predict_cla)][0])

            # print(class_indict[str(predict_cla)][0])
        # acc = count/lenth
    # print(class_indict)
    acc = accuracy_score(t_labels, p_labels)
    conmat = confusionMatrix(t_labels, p_labels, len(t_labels), len(class_indict))
    p, r, f1 = p_and_r(conmat)
    print("test_acc: {:.3}".format(acc))
    for i in range(2):
        print('class: %s  p: %.3f  r: %.3f  f1: %.3f' % (class_indict[str(i)], p[i], r[i], f1[i]))
            # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
            #                                              predict[predict_cla].numpy())
            # plt.title(print_res)
            # for i in range(len(predict)):
            #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
            #                                               predict[i].numpy()))
            # plt.show()


if __name__ == '__main__':
    main()
