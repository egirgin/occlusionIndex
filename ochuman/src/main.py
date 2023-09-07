#source ../../../../../smpl_venv/bin/activate
import sys

import cv2, json, os
from vis import *
import numpy as np
from utils import *
from operator import itemgetter
from occlusion_tools import occlusion_index, coco_subset, form_criterion, filter_by_criterion
from smpl_tools import * # import by name 
np.random.seed(39)

draw_and_save = False
criterion = "head_subset"


def main():

    prefix = "../images/"

    data = get_annos(split="train")

    #smpls = read_smpl(split="train")

    if not "gtSMPL" in os.listdir("."):
        process_smpl(read_smpl(split="train"))
        process_smpl(read_smpl(split="test"))

    num_frames = len(data["images"])

    criterion_mask = form_criterion(coco_subset[criterion])
    selected_imgs = []
    selected_occ_values = []

    occlusion_indices_list = []
    crowd_indices_list = []

    for frame_id in range(num_frames):
        img_path, img_masks, img_keypoints = ochuman_annotation(data, frame_id=frame_id)
        num_smpl, smpls = get_smpl(img_path)


        if img_path == None: # if there is missing mask or pose of a model
            continue

        if num_smpl == None:
            print("Frame {} No SMPL data, skipping...".format(frame_id))
            continue

        if len(img_keypoints) != num_smpl:
            print("Frame {} SMPL does not match keypoints... Skipping...".format(frame_id))
            continue


        # Calculate occlusion index
        occlusion_indices, occlusion_status, missing_indices, missing_keypoints_list = occlusion_index(masks=img_masks, keypoints=img_keypoints)

        frame_occlusion_index = np.mean(occlusion_indices)
        occlusion_indices_list.append([img_path, frame_occlusion_index])

        # body part occlusion

        occluded_by_criteria = filter_by_criterion(criterion=criterion_mask, occlusion_status=occlusion_status)

        if occluded_by_criteria:
            selected_imgs.append(img_path)
            selected_occ_values.append(frame_occlusion_index)

        # calculate crowd index
        crowd_indices, crowd_keypoints_list = crowd_index(keypoints=img_keypoints)

        frame_crowd_index = np.mean(crowd_indices)
        crowd_indices_list.append([img_path, frame_crowd_index])
        
        if draw_and_save and occluded_by_criteria:
            
            num_models = len(img_keypoints)

            # Draw elements
            colors = np.random.randint(255, size=(num_models, 3)).tolist()  # each person has its own color

            bboxes = []
            for model_id in range(num_models):
                bboxes.append(estimate_bbox(keypoints=img_keypoints[model_id]))


            img = read_img(prefix+img_path, show=False)

            # Draw occlusion image
            occlusion_img = draw_keypoints(img, keypoints=img_keypoints, occlusion_status=occlusion_status, colors=colors, draw_red=True, only_occluded=True)
            occlusion_img = draw_masks(occlusion_img, masks=img_masks, colors=colors)
            occlusion_img = draw_bboxes(occlusion_img, bboxes=bboxes, colors=colors, index_values=occlusion_indices) # TODO

            # Draw crowd image
            crowd_img = draw_keypoints(img, keypoints=img_keypoints, occlusion_status=occlusion_status, colors=colors, draw_red=True, only_occluded=False)
            crowd_img = draw_bboxes(crowd_img, bboxes=bboxes, colors=colors, index_values=crowd_indices) # TODO

            # Concat both
            concat_img = draw_sidebyside(img1=crowd_img, img2=occlusion_img, vis_scale=1, show=True)
            save_processed_img(concat_img, img_path)

    dump_sorted(filename_list=selected_imgs, index_list=selected_occ_values, subset_name=criterion)



if __name__ == '__main__':
    main()
    #            if cv2.pointPolygonTest(np_obj, (x, y), False) == 1:




