import numpy as np
import cv2
import os, json
from vis import show_img
from utils import read_img, estimate_bbox, occlusion_index, crowd_index, COCO_JOINT, joint_list
from enum import Enum

class RecordType(Enum):
    OCHuman = 0
    Agora = 1
    Dataset_3DPW = 2




class RecordHPS:
    def __init__(self, img_path, img_id=None, record_type:RecordType = None):
        self.record_type = record_type

        self.img_path = img_path
        self.img_id = img_id  # for ochuman, img id comes from annotation file
        self.filename = None
        self.process_img()

        self.num_models = None

        self.keypoints_2d_original = None
        self.keypoints2d = None
        self.keypoint_visibility = None

        self.bboxes = None

        self.img_mask = None
        self.instance_masks = None
        self.instance_colors = None

        self.keypoints3d = None

        self.occlusion_indices = None
        self.occlusion_status = None # occlusion status per joint (n_models, 19)

        self.missing_indices = None
        self.missing_status = None # missing status per joint (n_models, 19)


    def process_img(self):
        """
        should I remove .png/jpg/jpeg ?
        may lead to file not exists
        :return:
        """
        filename = self.img_path.split("/")[-1]

        self.filename = filename

    def img(self, scale=1, show=False):
        img = read_img(self.img_path, scale=scale)

        if show:
            show_img(img, window_name=self.img_id)

        return img


    def set_keypoints2d(self, keypoints_2d_original):
        self.keypoints_2d_original = keypoints_2d_original # keypoint format -> (x,y,c) 

        self.keypoints2d = keypoints_2d_original[:, :, :-1]

        self.num_missing_keypoints = np.sum(keypoints_2d_original[:, :, -1] == 0)

        self.num_models = len(self.keypoints2d)

        bboxes = []
        for model_id in range(self.num_models):
            bboxes.append(estimate_bbox(self.keypoints2d[model_id]))

        self.bboxes = np.array(bboxes)

    def set_masks(self, masks):
        if self.record_type == RecordType.OCHuman:
            self.instance_masks = masks
            self.instance_colors = np.random.randint(255, size=(self.num_models, 3)).tolist()

            if self.num_models != len(self.instance_masks):
                print("Number of models mismatch between keypoints and masks!!!")


        elif self.record_type == RecordType.Agora:
            pass
        else:
            pass

    def calculate_occlusion(self):
        self.occlusion_indices, self.occlusion_status, self.missing_indices, self.missing_status = occlusion_index(self.instance_masks, self.keypoints2d)
        self.crowd_indices, self.crowd_status = crowd_index(self.keypoints2d)
    