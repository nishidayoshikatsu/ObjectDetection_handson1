import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
import argparse
import random
import pickle as pkl

def prep_image(img, inp_dim):
    """
    画像をNNに入力するための形式に変更
    """
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()  # BGR -> RGB
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img):
    """
    baunding boxの描画
    """
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    print(c1, c2)
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img

def arg_parse():
    """
    コマンドライン引数の設定
    """

    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.25)      # しきい値
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--reso", dest = 'reso', help =
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "160", type = str)
    return parser.parse_args()


if __name__ == "__main__":
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "../yolov3.weights"

    # コマンドライン引数の取得
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()

    num_classes = 80
    bbox_attrs = 5 + num_classes

    model = Darknet(cfgfile)                    # モデルの定義
    model.load_weights(weightsfile)             # 学習済みの重みを読み込む

    model.net_info["height"] = args.reso        # 画像解像度
    inp_dim = int(model.net_info["height"])     # 画像の次元数を読み込み

    # assert文：条件式がFalseのときに例外
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # modelや変数をcudatensorとして定義することでGPU上で計算できる
    if CUDA:
        model.cuda()

    # modelの推論モード（学習モードはmodel.train()）
    model.eval()

    frames = 0
    start = time.time()
    img = cv2.imread("./dog.jpg")
    img = cv2.resize(img, (640, 480))

    if not img is None:
        img, orig_im, dim = prep_image(img, inp_dim)  # NNが読み込める形式にする
        if CUDA:
            img = img.cuda()
        # 出力の計算
        output = model(Variable(img), CUDA)
        output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

        output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim
        output[:,[1,3]] *= img.shape[1]
        output[:,[2,4]] *= img.shape[0]
        classes = load_classes("data/coco.names")   # クラス名の読み込み
        colors = pkl.load(open("pallete", "rb"))    # クラスごとの色の読み込み
        list(map(lambda x: write(x, orig_im), output))
        cv2.imshow("frame", orig_im)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("image not found.")