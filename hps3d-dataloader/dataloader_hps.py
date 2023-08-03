import numpy as np
import cv2
import os, json, sys
from enum import Enum
from record import RecordHPS, RecordType 
from utils import get_json, ochuman_annotation, COCO_JOINT, joint_list, COCO_SUBSET, form_criterion


class Dataset(Enum):
    OCHuman = 0

    Agora_Archviz = 1
    Agora_BrushifyForest = 2
    Agora_BurshifyGrasslands = 3
    Agora_Construction = 4
    Agora_Flowers = 5
    Agora_HDRI = 6

    Dataset_3DPW = 7

# we only consider train set, we may add validation and test splits as well
_datasets_paths = {
    "ochuman" : {
        "imgs" : "/home/emre/Documents/master/thesis/ochuman/OCHuman-20230218T083152Z-001/OCHuman/images/",
        "annotation" : "/home/emre/Documents/master/thesis/ochuman/OCHuman-20230218T083152Z-001/OCHuman/ochuman.json", # this is only trainset
        "smpl" : "/home/emre/Documents/master/thesis/ochuman/OCHuman-20230218T083152Z-001/OCHuman/smpl_labels/"
    },
    "agora" : {
        "train": "path",
        "val": "path",
        "test": "path",
        "annotation": "path",
        "smpl": "path"
    },
    "3dpw" : {
        "train": "path",
        "val": "path",
        "test": "path",
        "annotation": "path",
        "smpl": "path"
    }
}


class HPS_Dataloader:
    def __init__(self, dataset: Dataset, split="train"): # TODO: Split
        self.dataset = dataset
        self.img_list = None
        self.num_samples = None
        self.annos = None
        self.invalid_counter = 0
        self.occlusion_dict = {}


        if self.dataset == Dataset.OCHuman:
            self.process_ochuman()

        elif self.dataset == Dataset.Agora_Archviz:
            pass
        elif self.dataset == Dataset.Dataset_3DPW:
            pass
        else:
            pass

    def process_ochuman(self):

        img_folder_path = _datasets_paths["ochuman"]["imgs"]

        img_list = os.listdir(img_folder_path)
        img_list = list(map(lambda i: img_folder_path + i, img_list))

        self.img_list = img_list

        self.num_samples = len(self.img_list)

        anno_file_path = _datasets_paths["ochuman"]["annotation"]

        self.annos = get_json(anno_file_path)

        return img_list

    def _convert_record(self, index):

        record = None

        if self.dataset == Dataset.OCHuman:
            img_path, img_masks, img_keypoints = ochuman_annotation(self.annos, frame_id=index)

            if img_path == None:
                self.invalid_counter += 1
                return None

            img_folder_path = _datasets_paths["ochuman"]["imgs"]
            img_filename = img_folder_path + img_path
            
            # remember: img id in the annotation is different than the number on the filename!!
            record = RecordHPS(img_path=img_filename, img_id=index, record_type=RecordType.OCHuman)

            record.set_keypoints2d(img_keypoints)

            record.set_masks(img_masks)

            record.calculate_occlusion()


        elif self.dataset == Dataset.Agora_Archviz:
            pass
        elif self.dataset == Dataset.Dataset_3DPW:
            pass
        else:
            pass

        return record


    def __getitem__(self, index):
        record = self._convert_record(index)
        if record != None:
            self.occlusion_dict[index] = record.occlusion_status # accumulate occlusion values over fetch
        return record


    def sort_samples(self,criterion, drop_zeros=False):
        #sort samples based on a given weight of joints
        # make drop zeros true if you are using a subset as criterion

        criterion_dict = {}

        for idx in self.occlusion_dict.keys():
            avg_occlusion_frame = np.mean(self.occlusion_dict[idx], axis=0)

            occlusion_value = np.multiply(avg_occlusion_frame, criterion)
            occlusion_value = np.sum(occlusion_value)

            if drop_zeros and occlusion_value == 0:
                continue

            criterion_dict[idx] = occlusion_value

        decending_sorted = dict(sorted(criterion_dict.items(), key=lambda x:x[1]))

        return list(decending_sorted.keys()), list(decending_sorted.values()) 
    
    