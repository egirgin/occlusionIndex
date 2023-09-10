import os, sys, pickle, time, math, argparse
from visualize import *
from utils import *
from occlusion_tools import occlusion_index, filter_by_criterion, form_criterion, coco_subset

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
    
    subset_list = ["empty, head, torso, left_lower, left_upper, right_lower, right_upper", "right_body", "left_body", "upper_body", "lower_body"]

    ################################################################################################################
    parser = argparse.ArgumentParser(description="3dpw arg parser")

    # Check the paths below for different configs
    parser.add_argument('-s', '--scene', choices=scene_list + ['all'],  default="courtyard_basketball_00", help='Scene of 3DPW')
    parser.add_argument('-c', '--criteria', choices=subset_list,  default="empty", help='Subset of occlusion by body part')
    parser.add_argument('-d', '--save_results', action='store_true', default=True, help='Save visual results.')

    args = parser.parse_args()
    ################################################################################################################
    
    if args.scene != "all":
        scene_list = [args.scene]
    
    criterion = args.criteria + "_subset"
    draw = criterion == "empty_subset"

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
        
        criterion_mask = form_criterion(coco_subset[criterion])
        selected_imgs = []
        selected_occ_values = []
        selected_occ_status = []

        duration = []
        
        for frame_id in range(scene_length):
            start_time = time.time()

            image_filename = "image_{}.jpg".format(str(frame_id).zfill(5))
            img_path = dataset_config["img_folder_path"] + scene_name + "/" + image_filename

            remaining_secs = np.mean(duration)*(scene_length-frame_id)
            print("%{:.2f} Processing {}... ETA: {:.0f}mins {:.0f}secs".format(frame_id*100/scene_length, image_filename, remaining_secs//60, remaining_secs%60))
            
            try: # continue if that img does not exists
                img = read_im(img_path, show=False)
            except:
                print("Image does not exists")
                continue
            
            num_models = len(seq["poses"])

            keypoints2d = np.zeros((num_models, 24, 2))

            for model_id in range(len(seq["poses"])):

                keypoints2d[model_id] = estimate_from_3d(seq=seq, frame_id=frame_id, model_id=model_id)
            
            keypoints2d = smpl2coco(keypoints2d)
            
            intrinsic, R, t, extrinsic = get_cam_params(seq, frame_id)

            camera_matrices = {
                "intrinsics": intrinsic.tolist(),
                "extrinsics": extrinsic.tolist()
            }

            # get masks
            mask_path = dataset_config["mask_folder_path"] + scene_name + "/" + image_filename

            mask = read_im(mask_path, show=False)  

            #dilation_kernel = np.ones((3,3), np.uint8)

            #mask = cv2.dilate(mask, dilation_kernel, iterations=2)

            occlusion_indices, occlusion_status = occlusion_index(mask=mask, keypoints=keypoints2d, colormap=color_list)

            frame_occlusion_index = np.mean(occlusion_indices)

            # body part occlusion

            occluded_by_criteria = filter_by_criterion(criterion=criterion_mask, occlusion_status=occlusion_status)

            if occluded_by_criteria:
                selected_imgs.append(image_filename)
                selected_occ_values.append(frame_occlusion_index)
                selected_occ_status.append(["".join(frame_occ.astype(str)) for frame_occ in occlusion_status])

            # drawing

            if draw:
                
                # Draw elements
                colors = np.random.randint(255, size=(num_models, 3)).tolist()  # each person has its own color

                bboxes = []
                for model_id in range(num_models):
                    bboxes.append(approximate_bb(keypoints=keypoints2d[model_id]))

                # Draw occlusion image
                occlusion_img = draw_keypoints(img, keypoints=keypoints2d, occlusion_status=occlusion_status, only_occluded=False)
                occlusion_img = draw_bboxes(occlusion_img, bboxes=bboxes, colors=color_list, index_values=occlusion_indices)
                occlusion_img = draw_masks(occlusion_img, mask=mask)

                #show_im(occlusion_img)
                save_processed_img(occlusion_img, image_filename)

            end_time = time.time()

            duration.append(end_time-start_time)
            
        dump_sorted(filename_list=selected_imgs, index_list=selected_occ_values, occ_status=selected_occ_status, scene_name=scene_name, subset_name=criterion)
        
