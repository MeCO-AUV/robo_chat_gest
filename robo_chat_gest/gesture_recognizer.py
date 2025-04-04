#! /usr/bin/python3
"""
Interactive Robotics and Vision Lab
http://irvlab.cs.umn.edu/

Class for recognizing hand gestures from monocular images using two stage
system:
    The region of hands is first detected by YOLOv7-tiny. Then, the detected
    regions are cropped from the images and classified by a multitasking
    network which includes a transformer decoder.
"""

import os
import yaml
import cv2
import glob
import argparse
import numpy as np
import onnxruntime as ort

from .utils import get_max_preds, get_affine_transform, letterbox


class HandGestRecognizer:
    def __init__(self, **kwargs):
        cls_weight = kwargs.get('cls_weight')
        det_weight = kwargs.get('det_weight')
        self.det_img_size = kwargs.get('det_img_size')
        self.cls_img_size = kwargs.get('cls_img_size')

        # load hand gesture classifier
        self.classifier = self.load_onnx_model(cls_weight)

        # load hand detector
        self.detector = self.load_onnx_model(det_weight)

    def load_onnx_model(self, weight_path):
        if not os.path.exists(weight_path):
            assert False, "Model is not exist in {}".format(weight_path)

        # Create session options and set the number of threads explicitly
        options = ort.SessionOptions()
        # Specify the number of threads for intra-op parallelism
        options.intra_op_num_threads = 4
        # Specify the number of threads for inter-op parallelism
        options.inter_op_num_threads = 4

        session = ort.InferenceSession(
            weight_path,
            options,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        return session

    def convert(self, img):
        im = img.transpose((2, 0, 1))
        im = np.expand_dims(im, 0)
        im = np.ascontiguousarray(im)

        im = im.astype(np.float32)
        im /= 255
        return im

    def process_image_for_detection(self, ori_img):
        img = cv2.cvtColor(ori_img.copy(), cv2.COLOR_BGR2RGB)

        img, ratio, dwdh = letterbox(
            img, new_shape=self.det_img_size, auto=False)

        im = self.convert(img)

        return im, ratio, dwdh

    def process_image_for_classification(self, img, bbox):
        x1, y1, x2, y2 = bbox
        c = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
        origin_size = max(x2 - x1, y2 - y1) * 1.0
        trans = get_affine_transform(c, 1, 0, origin_size, self.cls_img_size)
        img = cv2.warpAffine(
            img,
            trans,
            (int(self.cls_img_size[0]), int(self.cls_img_size[1])),
            flags=cv2.INTER_LINEAR)

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        im = self.convert(img)

        return im

    def inference(self, frame):
        img, ratio, dwdh = self.process_image_for_detection(frame)

        outname = [i.name for i in self.detector.get_outputs()]
        inname = [i.name for i in self.detector.get_inputs()]
        inp = {inname[0]: img}

        outputs = self.detector.run(outname, inp)[0]

        if len(outputs):
            _, x0, y0, x1, y1, _, score = outputs[0]
            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh * 2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            box_width = box_height = \
                max(box[2] - box[0], box[3] - box[1]) * 1.0
            corner = [
                (box[0] + box[2] - box_width) / 2,
                (box[3] + box[1] - box_height) / 2]

            if score > 0.2:
                hand = self.process_image_for_classification(frame, box)

                inname = [i.name for i in self.classifier.get_inputs()]
                inp = {inname[0]: hand}
                label_pred, heatmap_pred = self.classifier.run(None, inp)

                h, w = heatmap_pred.shape[-2:]

                pred_label = np.argmax(label_pred[0])
                landmarks_pred, _ = get_max_preds(heatmap_pred)
                landmarks_pred = landmarks_pred.squeeze(0)
                landmarks_pred[:, 0] = \
                    landmarks_pred[:, 0] / w * box_width + corner[0]
                landmarks_pred[:, 1] = \
                    landmarks_pred[:, 1] / h * box_height + corner[1]

                landmarks_pred = landmarks_pred.astype(np.int32)

                return pred_label, landmarks_pred, box

        return -1, None, None
