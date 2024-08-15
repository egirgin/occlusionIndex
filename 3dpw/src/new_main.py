import os, sys, pickle, time, math, argparse
from utils import *
from utils_new import joint_names, occlusion_index, crowd_metric, color_list
from vis_new import read_im, show_im, draw_bboxes, draw_keypoints, draw_masks

if __name__ == '__main__':

    dataset_config = {
        "img_folder_path": "/home/tuba/Documents/emre/thesis/dataset/3dpw/imageFiles/",
        "mask_folder_path": "/home/tuba/Documents/emre/thesis/occlusionIndex/3dpw/masks/",
        "annotation_folder_path": "/home/tuba/Documents/emre/thesis/dataset/3dpw/sequenceFiles/sequenceFiles/",
    }

    scene_list = ['courtyard_warmWelcome_00', 'courtyard_captureSelfies_00', 
              'courtyard_dancing_01', 'courtyard_goodNews_00',
                'courtyard_giveDirections_00', 'courtyard_hug_00', 
                'courtyard_dancing_00', 'courtyard_basketball_00', 
              'courtyard_shakeHands_00', 'downtown_bar_00']
    
    other_scene_lists = [ # either no occlusion or neglectable occlusion 
        'downtown_cafe_00', 'downtown_arguing_00', 'downtown_bus_00', 'courtyard_rangeOfMotions_01',
        'courtyard_capoeira_00', 'courtyard_rangeOfMotions_00', 'courtyard_drinking_00', 'courtyard_arguing_00'
    ]

    joints = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', "left_hand", "right_hand"
    ]
    num_joints = 24

    ################################################################################################################
    parser = argparse.ArgumentParser(description="3dpw arg parser")

    # Check the paths below for different configs
    parser.add_argument('-s', '--scene', choices=scene_list + ['all'],  default="courtyard_basketball_00", help='Scene of 3DPW')

    args = parser.parse_args()
    ################################################################################################################

    draw = True # modify if desired

    #scene_list = ["courtyard_warmWelcome_00"] 

    for scene_name in scene_list:

        scene_path = dataset_config["img_folder_path"] + scene_name

        scene_length = len(os.listdir(scene_path))

        if scene_name+".pkl" in os.listdir(dataset_config["annotation_folder_path"] + "train"):
            seq = get_seq(dataset_config["annotation_folder_path"] + "train", seq_name=scene_name)
        
        elif scene_name+".pkl" in os.listdir(dataset_config["annotation_folder_path"] + "validation"):
            seq = get_seq(dataset_config["annotation_folder_path"] + "validation", seq_name=scene_name)

        elif scene_name+".pkl" in os.listdir(dataset_config["annotation_folder_path"] + "test"):
            seq = get_seq(dataset_config["annotation_folder_path"] + "test", seq_name=scene_name)
        
        else:
            print("Sequence {} could not be found!".format(scene_name))
            sys.exit()

        image_list = os.listdir(scene_path)

        duration = []

        for frame_id, image_name in enumerate(image_list):

            start_time = time.time()

            image_filename = "image_{}.jpg".format(str(frame_id).zfill(5))
            img_path = dataset_config["img_folder_path"] + scene_name + "/" + image_filename

            remaining_secs = np.mean(duration)*(scene_length-frame_id)
            print("%{:.2f} Processing {}'s {}... ETA: {:.0f}mins {:.0f}secs".format(frame_id*100/scene_length, scene_name, image_filename, remaining_secs//60, remaining_secs%60))
            
            try: # continue if that img does not exists
                img = read_im(img_path, show=False)
            except:
                print("Image does not exists")
                continue
            
            #################################################################################################################################33
            # calculate 2d locations
            num_models = len(seq["poses"])

            keypoints2d = np.zeros((num_models, num_joints, 2))

            for model_id in range(num_models):

                keypoints2d[model_id] = estimate_from_3d(seq=seq, frame_id=frame_id, model_id=model_id)
            
            intrinsic, R, t, extrinsic = get_cam_params(seq, frame_id)

            camera_matrices = {
                "intrinsics": intrinsic.tolist(),
                "extrinsics": extrinsic.tolist()
            }
            
            #################################################################################################################################
            # calculate occlusion index
            # get masks
            mask_path = dataset_config["mask_folder_path"] + scene_name + "/" + image_filename

            mask = read_im(mask_path, show=False)

            occluded_keypoints_list, truncated_keypoints_list = occlusion_index(mask=mask, keypoints=keypoints2d, colormap=color_list)

            #################################################################################################################################
            # calculate crowd index
            # get bboxes

            crowd_index = {
                "overall": 0.0
            }
            crowd_indices = []

            for model_id in range(num_models):
                top_left, bottom_right = approximate_bb(keypoints=keypoints2d[model_id])
                bbox = [top_left, bottom_right]
                current_crowd_index = crowd_metric(bbox=bbox, keypoints=keypoints2d, current_model_id=model_id)
                crowd_index[model_id] = current_crowd_index
                crowd_indices.append(current_crowd_index)

            crowd_index["overall"] = np.mean(crowd_indices)
                        
            #################################################################################################################################
            # collect statistics
            frame_occlusion_index = {}
            
            frame_overall_occlusion = np.mean(occluded_keypoints_list.astype(int))
            
            frame_occlusion_index["overall"] = frame_overall_occlusion

            per_model_occlusion = []

            for model_id in range(num_models):

                model_occlusion_index = {
                    "occlusion": {
                        "overall": np.mean(occluded_keypoints_list[model_id].astype(int))
                    },
                    "truncation": {
                        "overall": np.mean(truncated_keypoints_list[model_id].astype(int))
                    }
                }

                per_model_occlusion.append(model_occlusion_index["occlusion"]["overall"])

                for keypoint_id in range(num_joints):
                    model_occlusion_index["occlusion"][joints[keypoint_id]] = bool(occluded_keypoints_list[model_id][keypoint_id])
                    model_occlusion_index["truncation"][joints[keypoint_id]] = bool(truncated_keypoints_list[model_id][keypoint_id])

                frame_occlusion_index[model_id] = model_occlusion_index

            #################################################################################################################################
            # save results
            results_dir_oi = "./3dpw/occlusion_index/{}".format(scene_name)
            results_dir_ci = "./3dpw/crowd_index/{}".format(scene_name)

            """
            os.makedirs(results_dir_oi, exist_ok=True)
            os.makedirs(results_dir_ci, exist_ok=True)

            with open("{}/{}.json".format(results_dir_oi, image_filename[:-4]), "w+") as occlusion_filepointer:
                json.dump(frame_occlusion_index, occlusion_filepointer)

            with open("{}/{}.json".format(results_dir_ci, image_filename[:-4]), "w+") as crowd_filepointer:
                json.dump(crowd_index, crowd_filepointer)
            """
            #################################################################################################################################
            # drawing

            if draw:
                
                # Draw elements
                colors = np.random.randint(255, size=(num_models, 3)).tolist()  # each person has its own color

                bboxes = []
                for model_id in range(num_models):
                    bboxes.append(approximate_bb(keypoints=keypoints2d[model_id]))

                # Draw occlusion image
                occlusion_img = draw_masks(img.copy(), mask=mask)
                occlusion_img = draw_keypoints(occlusion_img, keypoints=keypoints2d, occlusion_index=frame_occlusion_index, kp_names=joints, only_occluded=False, colors=color_list)
                occlusion_img = draw_bboxes(occlusion_img, bboxes=bboxes, colors=color_list, index_values=per_model_occlusion)
                

                crowd_img = draw_keypoints(img.copy(), keypoints=keypoints2d, occlusion_index=frame_occlusion_index, kp_names=joints, only_occluded=False, colors=color_list)
                crowd_img = draw_bboxes(crowd_img, bboxes=bboxes, colors=color_list, index_values=crowd_indices)

                #show_im(occlusion_img)
                #show_im(crowd_img)

                cv2.imwrite("{}/{}".format(results_dir_oi, image_filename), occlusion_img)
                cv2.imwrite("{}/{}".format(results_dir_ci, image_filename), crowd_img)

            end_time = time.time()

            duration.append(end_time-start_time)
        
