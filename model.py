from config import *
from crnn import CRNNHandle
from angnet import  AngleNetHandle
from utils import draw_bbox, crop_rect, sorted_boxes, get_rotate_crop_image
from PIL import Image
import numpy as np
import cv2
import copy
from dbnet.dbnet_infer import DBNET
import time
import traceback

class  OcrHandle(object):
    def __init__(self):
        self.text_handle = DBNET(model_path, providers=providers)
        self.crnn_handle = CRNNHandle(crnn_model_path, providers=providers)
        if angle_detect:
            self.angle_handle = AngleNetHandle(angle_net_path, providers=providers)


    def crnnRecWithBox(self,im, boxes_list,score_list):
        """
        crnn模型，ocr识别
        @@model,
        @@converter,
        @@im:Array
        @@text_recs:text box
        @@ifIm:是否输出box对应的img

        """
        results = []
        boxes_list = sorted_boxes(np.array(boxes_list))

        line_imgs = []
        for index, (box, score) in enumerate(zip(boxes_list[:angle_detect_num], score_list[:angle_detect_num])):
            tmp_box = copy.deepcopy(box)
            partImg_array = get_rotate_crop_image(im, tmp_box.astype(np.float32))
            partImg = Image.fromarray(partImg_array).convert("RGB")
            line_imgs.append(partImg)

        angle_res = False
        if angle_detect:
            angle_res = self.angle_handle.predict_rbgs(line_imgs)

        count = 1
        for index, (box ,score) in enumerate(zip(boxes_list,score_list)):

            tmp_box = copy.deepcopy(box)
            partImg_array = get_rotate_crop_image(im, tmp_box.astype(np.float32))


            partImg = Image.fromarray(partImg_array).convert("RGB")

            if angle_detect and angle_res:
                partImg = partImg.rotate(180)


            if not is_rgb:
                partImg = partImg.convert('L')

            try:
                if is_rgb:
                    simPred = self.crnn_handle.predict_rbg(partImg)  ##识别的文本
                else:
                    simPred = self.crnn_handle.predict(partImg)  ##识别的文本
            except Exception as e:
                print(traceback.format_exc())
                continue

            if simPred.strip() != '':
                results.append([tmp_box,"{}、 ".format(count)+  simPred,score])
                count += 1

        return results


    def text_predict(self,img,short_size):
        boxes_list, score_list = self.text_handle.process(np.asarray(img).astype(np.uint8),short_size=short_size)
        result = self.crnnRecWithBox(np.array(img), boxes_list,score_list)

        return result


if __name__ == "__main__":
    print('-'*100)
    path = '1.png'
    providers = ['CPUExecutionProvider']
    # path = 'test_imgs/dotnet/OcrLiteOnnxCs.PNG'
    img = Image.open(path)
    # img = img.rotate(180)
    ocr = OcrHandle()
    size = min(img.size) // 32 * 32

    import time
    start = time.time()
    num = 1
    for i in range(num):
        result = ocr.text_predict(img, size)
    print(result)
    print(len(result))
    end = time.time()
    print('avg time', (end-start)/num)
