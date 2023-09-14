from visualize import *
from utils import *
from enum import Enum

def occlusion_index(masks, keypoints):

    num_models = len(keypoints)

    all_mask = read_im(masks[0], scale=1, show=False)

    dilation_kernel = np.ones((2,2), np.uint8)

    occluded_keypoints_list = np.zeros((num_models, 19), dtype=int) # 1 means occluded, 0 means visible 
    occlusion_indices = []
     
    for model_id in range(num_models): # this brings keypoints

        mask = read_im(masks[model_id + 1], scale=1, show=False)  # mask_files[0] shows all the people

        person_unique_color, all_mask_unique_color = get_unique_color(mask, all_mask)

        if person_unique_color == None: # sometimes it happens because full background mask exists in agora dataset
            # here we assume it is completely occluded, prob. true though
            occlusion_indices.append(1)
            occluded_keypoints_list[model_id] = np.ones(19, dtype=int)
            continue

        all_mask[all_mask == (0, 0, 255)] = 0 # make bg red to black

        dilated_all_mask = cv2.dilate(all_mask, dilation_kernel, iterations=0) # if needed

        n_occluded_keypoints, occlusion_status = occlusion_helper(all_mask=dilated_all_mask, keypoints=keypoints[model_id], model_color=all_mask_unique_color)

        occluded_keypoints_list[model_id] = occlusion_status

        occlusion_indices.append(n_occluded_keypoints / 19) # coco pose format has 19 kps

    return occlusion_indices, occluded_keypoints_list

def occlusion_helper(all_mask, keypoints, model_color):

    # keypoints: occluded
    # person color code: occluded

    neighbor_offset = 1

    occluded_keypoints = np.zeros(19, dtype=int) # 1 means occluded, 0 means visible 

    for kp_id, keypoint in enumerate(keypoints):

        pixels = [
            [int(np.floor(keypoint[1])), int(np.floor(keypoint[0]))],
            [int(np.floor(keypoint[1])) + neighbor_offset, int(np.floor(keypoint[0]))], # bottom
            [int(np.floor(keypoint[1])), int(np.floor(keypoint[0])) + neighbor_offset], # right
            [int(np.floor(keypoint[1])) + neighbor_offset, int(np.floor(keypoint[0])) + neighbor_offset], # right - bottom
            ]

        keypoint_pixel_values = []

        for pixel in pixels:
            try: # may be out of the canvas
                keypoint_pixel_values.append(all_mask[pixel[0], pixel[1]])
            except:
                continue

        if len(keypoint_pixel_values) == 0:
            continue

        #if list(keypoint_pixel_values[0]) == [0,0,255]: # drop background
        #    continue

        if np.mean(np.absolute(keypoint_pixel_values - model_color), axis=1).all() != 0.0 :  # if the pixel value isn't the same
            occluded_keypoints[kp_id] = 1


    return np.sum(occluded_keypoints), occluded_keypoints


class COCO_JOINT(Enum):
    RIGHT_SHOULDER = 0
    RIGHT_ELBOW = 1
    RIGHT_WRIST = 2
    
    LEFT_SHOULDER = 3
    LEFT_ELBOW = 4
    LEFT_WRIST = 5

    RIGHT_HIP = 6
    RIGHT_KNEE = 7
    RIGHT_ANKLE = 8

    LEFT_HIP = 9
    LEFT_KNEE = 10
    LEFT_ANKLE = 11
    
    HEAD = 12
    NECK = 13

    RIGHT_EAR = 14
    LEFT_EAR = 15

    NOSE = 16

    RIGHT_EYE = 17
    LEFT_EYE = 18
    
class COCO_SUBSET(Enum):

    right_upper_subset = [COCO_JOINT.RIGHT_SHOULDER, COCO_JOINT.RIGHT_ELBOW, COCO_JOINT.RIGHT_WRIST]
    left_upper_subset = [COCO_JOINT.LEFT_SHOULDER, COCO_JOINT.LEFT_ELBOW, COCO_JOINT.LEFT_WRIST]

    right_lower_subset = [COCO_JOINT.RIGHT_HIP, COCO_JOINT.RIGHT_KNEE, COCO_JOINT.RIGHT_ANKLE]
    left_lower_subset = [COCO_JOINT.LEFT_HIP, COCO_JOINT.LEFT_KNEE, COCO_JOINT.LEFT_ANKLE]

    head_subset = [COCO_JOINT.HEAD, COCO_JOINT.NECK, COCO_JOINT.NOSE]
    torso_subset = [COCO_JOINT.RIGHT_SHOULDER, COCO_JOINT.RIGHT_HIP, COCO_JOINT.LEFT_SHOULDER, COCO_JOINT.LEFT_HIP]

coco_subset = {
    "right_upper_subset" : [COCO_JOINT.RIGHT_SHOULDER, COCO_JOINT.RIGHT_ELBOW, COCO_JOINT.RIGHT_WRIST],
    "left_upper_subset" : [COCO_JOINT.LEFT_SHOULDER, COCO_JOINT.LEFT_ELBOW, COCO_JOINT.LEFT_WRIST],

    "right_lower_subset" : [COCO_JOINT.RIGHT_HIP, COCO_JOINT.RIGHT_KNEE, COCO_JOINT.RIGHT_ANKLE],
    "left_lower_subset" : [COCO_JOINT.LEFT_HIP, COCO_JOINT.LEFT_KNEE, COCO_JOINT.LEFT_ANKLE],

    "head_subset" : [COCO_JOINT.HEAD, COCO_JOINT.NECK, COCO_JOINT.NOSE],
    "torso_subset" : [COCO_JOINT.RIGHT_SHOULDER, COCO_JOINT.RIGHT_HIP, COCO_JOINT.LEFT_SHOULDER, COCO_JOINT.LEFT_HIP],

    "left_body_subset" : COCO_SUBSET.left_upper_subset.value + COCO_SUBSET.left_lower_subset.value,
    "right_body_subset" : COCO_SUBSET.right_upper_subset.value + COCO_SUBSET.right_lower_subset.value,
    "upper_body_subset" : COCO_SUBSET.right_upper_subset.value + COCO_SUBSET.left_upper_subset.value,
    "lower_body_subset" : COCO_SUBSET.left_upper_subset.value + COCO_SUBSET.left_lower_subset.value,

    "empty_subset" : []
}

def form_criterion(subset):
    criterion = np.zeros(19, dtype=bool)

    for i in range(19):
        for joint in subset:
            criterion[joint.value] = True

    return criterion

joint_list =  ['right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder',
                'left_elbow', 'left_wrist', 'right_hip', 'right_knee', 'right_ankle', 
                'left_hip', 'left_knee', 'left_ankle', 'head', 'neck', 'right_ear', 
                'left_ear', 'nose', 'right_eye', 'left_eye'] 

def filter_by_criterion(criterion, occlusion_status):
    for model_occlusion in occlusion_status:
        
        if np.all(model_occlusion[criterion]):
            #print(model_occlusion[criterion])
            return True
    
    return False










def crowd_metric(bbox, keypoints):

    intersection_keypoints = []

    count = 0
    for keypoint in keypoints:
        if bbox[0][0] < keypoint[0] < bbox[1][0] and bbox[0][1] < keypoint[1] < bbox[1][1]:
            count += 1
            intersection_keypoints.append([keypoint[0], keypoint[1]])

    return count, intersection_keypoints
