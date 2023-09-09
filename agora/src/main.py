import os, sys, pickle, time, math
from visualize import *
from utils import *
from occlusion_tools import occlusion_index, filter_by_criterion, form_criterion, coco_subset

if __name__ == '__main__':

    dataset_config = {
        "img_folder_path": "/home/tuba/Documents/emre/thesis/dataset/agora/data/imgs/validation_images_1280x720/validation",
        #"smpl_folder_path": "/home/tuba/Documents/emre/thesis/dataset/agora/smpl/",
        "mask_folder_path": "/home/tuba/Documents/emre/thesis/dataset/agora/data/masks/validation_masks_1280x720/validation",
        "cam_folder_path": "/home/tuba/Documents/emre/thesis/dataset/agora/data/cam/validation_annos",
    }


    """
    scene options:
        archviz, #259
        brushifyforest, #258 
        brushifygrasslands, #259
        construction, # 256
        flowers, # 259
        hdri_50mm  # 258
    """

    if len(sys.argv) > 1:
        scene = sys.argv[1]
    else:
        scene = "archviz"

    if scene == "archviz":
        num_people = "5_10"
    else:
        num_people = "5_15"


    img_config= {
        "split": "validationset",
        "vendor": "renderpeople",
        "dataset": "bfh",
        "scene": scene,
        "num_people": num_people,
        "cam": "cam02",
        "resolution": "1280x720"
    }

    if img_config["scene"] == "archviz":
        scene_length = 259
    elif img_config["scene"] == "brushifyforest":
        scene_length = 258
    elif img_config["scene"] == "brushifygrasslands":
        scene_length = 259
    elif img_config["scene"] == "construction":
        scene_length = 256
    elif img_config["scene"] == "flowers":
        scene_length = 259
    elif img_config["scene"] == "hdri_50mm":
        scene_length = 258
    else:
        sys.exit()


    # flags
    scale = 1
    draw = False
    if len(sys.argv) > 2:
        criterion = sys.argv[2]
    else:
        criterion = "head_subset"

    criterion_mask = form_criterion(coco_subset[criterion])
    selected_imgs = []
    selected_occ_values = []
    selected_occ_status = []

    duration = []
    
    for frame_id in range(scene_length):
        start_time = time.time()
        img_path, img_filename = construct_img_path(dataset_config, img_config, frame_id=frame_id)
        remaining_secs = np.mean(duration)*(scene_length-frame_id)
        print("%{:.2f} Processing {}... ETA: {:.0f}mins {:.0f}secs".format(frame_id*100/scene_length, img_filename, remaining_secs//60, remaining_secs%60))

        try: # continue if that img does not exists
            img = read_im(img_path, scale, show=False)
        except:
            print("Image does not exists")
            continue
        
        annotation = read_anno(dataset_config=dataset_config, img_filename=img_filename)
        keypoints2d = np.array(annotation["keypoints2d"]) * (720/2160) # from 4K to HD
        keypoints2d = agora2coco(keypoints2d)
        
        # get masks
        mask_files = construct_mask_path(dataset_config, img_config, frame_id=frame_id)

        all_mask = read_im(mask_files[0], scale, show=False)

        assert img.shape == all_mask.shape, "mask & img shape does not match"     

        # check consistency
        mask_num_models = len(mask_files) - 1

        num_models = keypoints2d.shape[0]

        assert mask_num_models == num_models,  "mask and keypoint model number does not match"

        occlusion_indices, occlusion_status = occlusion_index(masks=mask_files, keypoints=keypoints2d)

        frame_occlusion_index = np.mean(occlusion_indices)

        # body part occlusion

        occluded_by_criteria = filter_by_criterion(criterion=criterion_mask, occlusion_status=occlusion_status)

        if occluded_by_criteria:
            selected_imgs.append(img_filename)
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
            occlusion_img = draw_keypoints(img, keypoints=keypoints2d, occlusion_status=occlusion_status, only_occluded=True)
            occlusion_img = draw_bboxes(occlusion_img, bboxes=bboxes, colors=colors, index_values=occlusion_indices)
            occlusion_img = draw_masks(occlusion_img, mask=all_mask)

            #show_im(occlusion_img)
            save_processed_img(occlusion_img, img_filename)

        end_time = time.time()

        duration.append(end_time-start_time)

    dump_sorted(filename_list=selected_imgs, index_list=selected_occ_values, occ_status=selected_occ_status, scene_name=img_config["scene"], subset_name=criterion)
    
