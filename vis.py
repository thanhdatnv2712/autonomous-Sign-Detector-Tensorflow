import cv2
import tensorflow as tf
import keras
import os
import time
import argparse
import glob
from tqdm import tqdm
from lib.core.api.face_detector import FaceDetector, SignRecognition
# from lib.core.api.sign_recognition import SignRecognition

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def image_demo(data_dir):
    recognition = SignRecognition("/home/ubuntu/seg/autonomous-Sign-Detector/checkpoints/best_ckpt.h5")
    detector = FaceDetector(args.model)

    new_dir= data_dir + "_result"
    files = glob.glob(data_dir + '/*')
    files.sort(key=os.path.getmtime)
    for pic in tqdm(files):
        if pic.endswith('jpg') or pic.endswith("png"):
            img = cv2.imread(pic)

            img_show = img.copy()

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pred = detector(img,0.4)


            for i, pred_i in enumerate(pred):
                if pred_i.shape[0] > 0:
                    for bbox in pred_i:
                        cv2.rectangle(img_show, (int(bbox[0]), int(bbox[1])),
                                    (int(bbox[2]), int(bbox[3])), (255, 0, 255), 2)
                        cv2.putText(img_show, str(i), (int(bbox[0]), int(bbox[1])), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            newpath= pic.replace(data_dir, new_dir)
            dn = os.path.dirname(newpath)
            if not os.path.exists(dn):
                os.makedirs(dn)
            cv2.imwrite(newpath, img_show)
            # cv2.namedWindow('res',0)
            # cv2.imshow('res',img_show)
            # k = cv2.waitKey(0)
            # if k == 110:
            #     continue
            # else:
            #     break

def demo():
    args.model
    detector = FaceDetector(args.model)
    # files = glob.glob(data_dir + '/*')
    # files.sort(key=os.path.getmtime)
    files= ["/home/ubuntu/seg/TAD16K/test_3/32871.jpg"]
    for pic in files:
        if pic.endswith('jpg'):
            img = cv2.imread(pic)

            img_show = img.copy()

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pred = detector(img,0.4)
            print('pred ', pred.shape, pred)

            for i, pred_i in enumerate(pred):
                if pred_i.shape[0] > 0:
                    for bbox in pred_i:
                        cv2.rectangle(img_show, (int(bbox[0]), int(bbox[1])),
                                    (int(bbox[2]), int(bbox[3])), (255, 0, 255), 2)
                        cv2.putText(img_show, str(i), (int(bbox[0]), int(bbox[1])), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            cv2.imwrite("test.png", img_show)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start train.')

    parser.add_argument('--model', dest='model', type=str, default="/home/ubuntu/seg/autonomous-car-2020-sign-detection/model/epoch_556_val_loss0.736055", help='the model to use')
    parser.add_argument('--img_dir', dest='img_dir', type=str, default="/home/ubuntu/seg/SignDataset/test", help='image directory')
    
    args = parser.parse_args()

    image_demo(args.img_dir)
    # demo()