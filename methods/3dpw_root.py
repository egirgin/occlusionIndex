import os, sys, argparse, time
import numpy as np
import romp, bev, cv2, json
import pandas as pd

from utils import annotation_3dpw, compute_error, coco2smpl
from error_metric import compute_mpjpe, compute_shape_error

# path check is a must

global dataset, dataset_path,  imglist_file, results_dir, save_results, model_type


parser = argparse.ArgumentParser(description="3dpw arg parser")

# Check the paths below for different configs
parser.add_argument('-m', '--model', choices=["romp", "bev"], default="romp", help='Model to be used.')
parser.add_argument('-d', '--save_results', action='store_true', default=False, help='Save results of model.')

args = parser.parse_args()

################################################################################################################
model_type = args.model

dataset = "3dpw" 

dataset_parent_folder_path = "/home/tuba/Documents/emre/thesis/dataset/3dpw/"

save_results = args.save_results

################################################################################################################
def get_img_list(selected_list_path):
    with open(selected_list_path, "r") as img_list:
        input_list = img_list.read().splitlines()

    filelist = []
    occlusion_mask_list = []
    for frame in input_list:
        filelist.append(frame.split()[0])
        occlusion_status = eval(frame.split("#")[-1])
        occlusion_status = np.array([[int(char) for char in model_occ] for model_occ in occlusion_status], dtype=bool)
        occlusion_mask_list.append(occlusion_status)

    return filelist, occlusion_mask_list

def read_occlusion_tables(occlusion_path, occlusion_mask_path):

    occlusion_table = pd.read_csv(occlusion_path).to_dict()
    occlusion_mask_table = pd.read_csv(occlusion_mask_path).to_dict()

    return occlusion_table, occlusion_mask_table


if __name__ == '__main__':
    
    ################################MODELS#######################################################
    if model_type == "romp":
        settings = romp.main.default_settings
        # settings is just a argparse Namespace. To change it, for instance, you can change mode via
        # settings.mode='video'
        settings.render_mesh = True
        settings.show = False
        settings.show_items = "mesh,mesh_bird_view"
        #settings.save_path = "."
        #settings.save_video = True
        if not save_results:
            settings.render_mesh = False

        model = romp.ROMP(settings)
        print(romp.main.default_settings)

    if model_type == "bev":
        settings = bev.main.default_settings
        # settings is just a argparse Namespace. To change it, for instance, you can change mode via
        # settings.mode='video'
        settings.render_mesh = True
        settings.show = False
        settings.show_items = "mesh,mesh_bird_view"
        #settings.save_path = "."
        #settings.save_video = True
        if not save_results:
            settings.render_mesh = False
        model = bev.BEV(settings)
        print(bev.main.default_settings)

    ################################ERRORS#######################################################


    cluster_mpjpe = {}
    cluster_pa_mpjpe  = {}
    for i in range(7):
        cluster_mpjpe[i] = []
        cluster_pa_mpjpe[i] = []

    ####################RUN########################################################################
    list_path = "./3dpw/selected_frames/self_occlusion_list.txt"

    with open(list_path, "r") as imglist_file:
        img_list = imglist_file.readlines()

    length = len(img_list)
    duration = []

    for idx, record in enumerate(img_list):

        start_time = time.time()

        cluster_id, image_path, model_id = record.split()
        cluster_id = int(cluster_id)
        model_id = int(model_id)

        scene_name = image_path.split("/")[0]
        im_filename = image_path.split("/")[1]

        results_dir = "./methods/{}_results/{}/{}/".format(model_type, dataset, scene_name)
        os.makedirs(results_dir, exist_ok=True)


        dataset_imgs_path = dataset_parent_folder_path + "imageFiles/{}/".format(scene_name)
        dataset_annotations_path = dataset_parent_folder_path + "sequenceFiles/sequenceFiles/"
        
        # get GTs
        thetas_gt, betas_gt = annotation_3dpw(annotation_filepath=dataset_annotations_path, scene_name=scene_name, im_filename =im_filename)
        thetas_gt = thetas_gt[model_id]
        betas_gt = betas_gt[model_id]
        
        # predict
        img = cv2.imread(dataset_imgs_path + im_filename)
        outputs = model(img)  # please note that we take the input image in BGR format (cv2.imread).
        
        if outputs == None:
            empty_image = np.zeros_like(img)
            thetas_pred = np.zeros((1,24,3))
            betas_pred = np.zeros((1,10))
        else:
            thetas_pred = outputs["smpl_thetas"].reshape(-1, 24, 3)
            betas_pred = outputs["smpl_betas"][:, :10].reshape(-1, 10)
        
        roots_pred = thetas_pred[:, 0]

        # Calculate the Euclidean distances between the new point and all points in the list
        distances = np.linalg.norm(roots_pred - thetas_gt[0], axis=1)

        # Find the index of the closest point
        closest_pred_id = np.argmin(distances)
        
        thetas_pred = thetas_pred[closest_pred_id]
        betas_pred = betas_pred[closest_pred_id]

        occlusion_mask = np.ones((1, 24), dtype=bool) # not care

        pose_coeff=0.9 
        shape_coeff=0.1

        mpjpe, pa_mpjpe, _, _ = compute_mpjpe(thetas_pred.reshape(1, 24, 3), thetas_gt.reshape(1, 24, 3), mask=occlusion_mask)

        shape_error = compute_shape_error(betas_pred.reshape(1, 10), betas_gt.reshape(1, 10))

        total_error = pose_coeff*mpjpe + shape_coeff*shape_error

        total_error_pa = pose_coeff*pa_mpjpe + shape_coeff*shape_error

        cluster_mpjpe[cluster_id].append(total_error)
        cluster_pa_mpjpe[cluster_id].append(total_error_pa)
        
        if save_results:
            if outputs == None:
                cv2.imwrite("{}/{}_{}".format(results_dir, scene_name, im_filename), empty_image)
            else:
                cv2.imwrite("{}/{}".format(results_dir, im_filename), outputs["rendered_image"])

        remaining_secs = np.mean(duration)*(length-idx)
        if idx%1000 == 0:
            print("%{:.2f} ETA: {:.0f}mins {:.0f}secs".format(idx*100/length, remaining_secs//60, remaining_secs%60))

        end_time = time.time()

        duration.append(end_time-start_time)

    with open("./methods/{}_results/3dpw/root_poses.txt".format(model_type), "w+") as result_file: 
        for cluster_id in range(7):
            cluster_mpjpe_avg = np.mean(cluster_mpjpe[cluster_id])
            cluster_pa_mpjpe_avg = np.mean(cluster_pa_mpjpe[cluster_id])

            s_mpjpe = "MPJPE for Cluster {}: {}".format(cluster_id, cluster_mpjpe_avg)
            s_pa_mpjpe = "PA_MPJPE for Cluster {}: {}".format(cluster_id, cluster_pa_mpjpe_avg)
            
            result_file.write(s_mpjpe + "\n")
            result_file.write(s_pa_mpjpe + "\n")

            print(s_mpjpe)
            print(s_pa_mpjpe)
        
    
    
    print(["{}".format(np.mean(cluster_mpjpe[i])) for i in range(7)])
    print(["{}".format(np.mean(cluster_pa_mpjpe[i])) for i in range(7)])

        

