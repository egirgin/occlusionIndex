#source ../../../../../smpl_venv/bin/activate
import sys

import cv2, json, os
from vis import *
import numpy as np
from utils import *
from operator import itemgetter
from smpl_np import *

def occlusion_index(masks, keypoints):
    num_models = len(keypoints)

    occlusion_indices = []

    occluded_keypoints_list = []

    for model_id in range(num_models): # this brings keypoints
        n_occluded_keypoints = 0
        occluded_keypoints = []
        for other_model_id in range(num_models): # this brings masks
            if model_id == other_model_id:
                continue

            for kp_id, kp in enumerate(keypoints[model_id]):
                for mask in masks[other_model_id]:
                    mask = np.array(mask, np.int32)
                    mask = mask.reshape((-1,2))
                    mask = mask.reshape((-1, 1, 2))
                    if cv2.pointPolygonTest(mask, (int(kp[0]), int(kp[1])), False) == 1:
                        n_occluded_keypoints += 1
                        occluded_keypoints.append(kp_id)

        occlusion_indices.append(n_occluded_keypoints / 19) # coco format has 19 keypoints
        occluded_keypoints_list.append(occluded_keypoints)

    return occlusion_indices, occluded_keypoints_list

def estimate_bbox(keypoints):
    x_offset = 20
    y_offset = 20

    nonzero_keypoints = keypoints[np.nonzero(keypoints)[0]]

    left_top = [int(nonzero_keypoints[:, 0].min() - x_offset), int(nonzero_keypoints[:, 1].min() - y_offset)]

    right_bottom = [int(keypoints[:, 0].max() + x_offset), int(keypoints[:, 1].max() + y_offset)]

    return np.array([left_top, right_bottom])

def crowd_index(keypoints):
    num_models = len(keypoints)

    crowd_indices = [] # holds values per person

    crowd_keypoints_list = [] # holds the list of all crowd keypoints

    for model_id in range(num_models):
        bbox = estimate_bbox(keypoints[model_id])

        crowd_keypoints = []

        for other_model_id in range(num_models):
            if model_id == other_model_id:
                continue

            for kp_id, kp in enumerate(keypoints[other_model_id]):
                left_top = bbox[0]
                right_bottom = bbox[1]

                if right_bottom[0] > kp[0] > left_top[0] and right_bottom[1] > kp[1] > left_top[1]:
                    crowd_keypoints.append(kp_id)

        crowd_indices.append(len(crowd_keypoints) / 19) # coco format has 19 keypoints
        crowd_keypoints_list.append(crowd_keypoints)

    return crowd_indices, crowd_keypoints_list

def dump_occlusion_indices(occlusion_indices, crowd_indices):

    sorted_index = sorted(occlusion_indices, key=itemgetter(1), reverse=True)

    with open("result_ochuman.txt", "w+") as result_file:
        for i, result in enumerate(sorted_index):
            result_file.write(
                "{} -> OI: {}\n".format(str(result[0]), result[1])
            )

    with open("sorted_occlusion_list.txt", "w+") as result_file:
        for result in sorted_index:
            result_file.write(
                "./ochuman/{}\n".format(str(result[0]))
            )

def dump_first_k(index, k=10, filename="occlusion_index"):
    os.makedirs("subdatasets", exist_ok=True)
    sorted_index = sorted(index, key=itemgetter(1), reverse=True)

    with open("./subdatasets/" + filename+"_{}.txt".format(k), "w+") as dump_file:
        for result in sorted_index[:k]:
            dump_file.write(
                "./ochuman/{}\n".format(str(result[0]))
            )


def main():

    prefix = "../images/"

    data = get_annos(split="train")

    #smpls = read_smpl(split="train")

    if not "gtSMPL" in os.listdir("."):
        process_smpl(read_smpl(split="train"))
        process_smpl(read_smpl(split="test"))

    num_frames = len(data["images"])


    occlusion_indices_list = []
    crowd_indices_list = []

    for frame_id in range(num_frames):
        img_path, img_masks, img_keypoints = get_frame_annos(data, frame_id=frame_id)
        num_smpl, smpls = get_smpl(img_path)


        if img_path == None:
            continue

        if num_smpl == None:
            print("Frame {} No SMPL data, skipping...".format(frame_id))
            continue

        if len(img_keypoints) != num_smpl:
            print("Frame {} SMPL does not match keypoints... Skipping...".format(frame_id))
            continue

        num_models = len(img_keypoints)

        img = read_img(prefix+img_path, show=False)

        # Calculate occlusion index
        occlusion_indices, occluded_keypoints_list = occlusion_index(masks=img_masks, keypoints=img_keypoints)

        frame_occlusion_index = np.mean(occlusion_indices)
        occlusion_indices_list.append([img_path, frame_occlusion_index])
        # calculate crowd index
        crowd_indices, crowd_keypoints_list = crowd_index(keypoints=img_keypoints)

        frame_crowd_index = np.mean(crowd_indices)

        crowd_indices_list.append([img_path, frame_crowd_index])

        # Draw elements
        colors = np.random.randint(255, size=(num_models, 3)).tolist()  # each person has its own color

        bboxes = []
        for model_id in range(num_models):
            bboxes.append(estimate_bbox(keypoints=img_keypoints[model_id]))

        #drawed_img = draw_mask_keypoint(img, img_masks, img_keypoints, occlusion_indices, occluded_keypoints=None) # to debug make occluded_keypoints_list

        # Draw occlusion image
        occlusion_img = draw_keypoints(img, keypoints=img_keypoints, colors=colors, occluded_keypoints=None)
        occlusion_img = draw_masks(occlusion_img, masks=img_masks, colors=colors)
        occlusion_img = draw_bboxes(occlusion_img, bboxes=bboxes, colors=colors, index_values=occlusion_indices)

        # Draw crowd image

        crowd_img = draw_keypoints(img, keypoints=img_keypoints, colors=colors, occluded_keypoints=None)
        crowd_img = draw_bboxes(crowd_img, bboxes=bboxes, colors=colors, index_values=crowd_indices)

        #drawed_img = draw_keypoints(img, keypoints=img_keypoints, colors=colors, occluded_keypoints=None) # set keypoint list if needed
        #drawed_img = draw_masks(img, masks=img_masks, colors=colors)
        #drawed_img = draw_bboxes(img, bboxes=bboxes, colors=colors, index_values=occlusion_indices) # for occlusion index
        #drawed_img = draw_bboxes(img, bboxes=bboxes, colors=colors, index_values=crowd_indices) # for crowd index

        #save_processed_img(drawed_img, frame_id)
        #show_img(drawed_img, frame_id)

        concat_img = draw_sidebyside(img1=crowd_img, img2=occlusion_img, vis_scale=1, show=False)
        save_processed_img(concat_img, img_path)

    dump_occlusion_indices(occlusion_indices_list, crowd_indices_list)

    for k in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        dump_first_k(occlusion_indices_list, k=k)
        dump_first_k(crowd_indices_list, k=k, filename="crowd_index")




if __name__ == '__main__':
    main()
    #            if cv2.pointPolygonTest(np_obj, (x, y), False) == 1:




