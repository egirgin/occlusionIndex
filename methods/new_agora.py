import os, sys, argparse, time
import numpy as np
import romp, bev, cv2, json
import pandas as pd
from scipy.spatial.transform import Rotation as R

from utils import annotation_3dpw, annotation_agora, compute_error, coco2smpl, computer_error_simple, computer_error_simple_joint
from error_metric import compute_similarity_transform



# path check is a must

global dataset, dataset_path, sublist, imglist_file, results_dir, save_results, model_type

scene_list = ['hdri_50mm', 'archviz']

parser = argparse.ArgumentParser(description="agora arg parser")

# Check the paths below for different configs
parser.add_argument('-m', '--model', choices=["romp", "bev"], default="romp", help='Model to be used.')
parser.add_argument('-d', '--save_results', action='store_true', default=True, help='Save results of model.')

args = parser.parse_args()
experiment_name = "14_dec"

################################################################################################################
model_type = args.model

dataset = "agora" 
dataset_parent_folder_path = "/home/tuba/Documents/emre/thesis/dataset/agora/data/"
dataset_imgs_path = dataset_parent_folder_path + "imgs/validation_images_1280x720/validation/"
dataset_annotations_path = dataset_parent_folder_path + "cam/validation_annos/"
dataset_smpl_path = dataset_parent_folder_path + "ground_truth/"

save_results = False # args.save_results

################################################################################################################
def draw_keypoints(image, keypoints):

    image_height, image_width = image.shape[:2]

    num_models = len(keypoints)

    for model_id in range(num_models):
        for kp_id, kp in enumerate(keypoints[model_id]):
            image = cv2.circle(image, (int(kp[0]), int(kp[1])), 1, (0, 0, 255), 2)
    return image
################################################################################################################
if __name__ == '__main__':
    start = time.time()
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
        settings.root_align = False
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
        settings.root_align = False
        if not save_results:
            settings.render_mesh = False
        model = bev.BEV(settings)
        print(bev.main.default_settings)

    ################################ERRORS#######################################################
    results_dir = "./methods/{}_results/{}/".format(model_type, dataset)
    os.makedirs(results_dir, exist_ok=True)

    mpjpe = {}
    pa_mpjpe = {}

    ####################RUN########################################################################
    #scene_list = [hdri_50mm] # archviz

    for scene_idx, scene_name in enumerate(scene_list):

        os.makedirs(results_dir + "/" + scene_name, exist_ok=True)
        print("Testing {} on {}'s {} scene...".format(model_type, dataset, scene_name))

        scene_img_list = [img_filename for img_filename in os.listdir(dataset_imgs_path) if scene_name in img_filename ]
        scene_length = len(scene_img_list)

        ############### ERROR PER SCENE###########################################################
        scene_mpjpe = {}
        scene_pa_mpjpe = {}   

        #######################################RUN OVER SAMPLES####################################
        #scene_img_list = ["image_00741.jpg"] 
        
        for i, im_filename in enumerate(scene_img_list): # first two are index and subset order
            print("Scene: {} | Progress: {:.2f}".format(scene_name, i*100/len(scene_img_list)))
            try: # continue if that img does not exists
                img = cv2.imread(dataset_imgs_path + im_filename)
                image_height, image_width = img.shape[:2]
            except:
                print("Image does not exists")
                continue
            im_filename = im_filename[:-13]

            ###############################################################################################
            # get GTs
            with open(dataset_annotations_path + im_filename + ".json", "r") as annotation_file:
                annotation = json.load(annotation_file)

            im_filename += ".png"

            smpl_paths = annotation["smpl_path"]
            location = annotation["location"]
            num_models = len(smpl_paths)

            imgWidth = image_width
            imgHeight = image_height

            thetas_gt, betas_gt, keypoints2d_gt = annotation_agora(dataset_smpl_path, smpl_paths,
                                                                   num_models, location, 
                                                                   imgHeight, imgWidth,
                                                                   scene=scene_name)
            
            ###############################################################################################
            # predict
            outputs = model(img)  # please note that we take the input image in BGR format (cv2.imread).
            
            if outputs == None: # If could not find any human
                empty_image = np.zeros_like(img)
                thetas_pred = np.zeros((1,24,3))
                betas_pred = np.zeros((1,10))
                keypoints2d_pred = np.zeros(1,24,2)
            else:
                thetas_pred = outputs["smpl_thetas"].reshape(-1, 24, 3)
                betas_pred = outputs["smpl_betas"][:, :10].reshape(-1, 10)
                keypoints2d_pred = outputs["pj2d_org"][:, :24, :]
            ###############################################################################################
            # Calculate error
            frame_mpjpe, frame_pa_mpjpe = computer_error_simple_joint(thetas_gt=thetas_gt, thetas_pred=thetas_pred, keypoints2d_gt=keypoints2d_gt, keypoints2d_pred=keypoints2d_pred)
            ###############################################################################################
            # Record frame error
            scene_mpjpe[im_filename] = {}
            scene_pa_mpjpe[im_filename] = {}
            """
            {
                scene_name:
                    image_name:
                        model_id:
                            overall:
                            joint0:
                            ...
                        overall:
                        n_preds:
            }
            
            """

            for model_id in range(num_models):
                scene_mpjpe[im_filename][model_id] = {
                    "overall": 0.0
                }
                scene_pa_mpjpe[im_filename][model_id] = {
                    "overall": 0.0
                }
                for joint_id in range(24):
                    if frame_mpjpe[model_id][joint_id] == 9999:
                        #print("Could not find the human at {} {}!".format(scene_name, im_filename))
                        pass

                    scene_mpjpe[im_filename][model_id][joint_id] = frame_mpjpe[model_id][joint_id] 
                    scene_pa_mpjpe[im_filename][model_id][joint_id] = frame_pa_mpjpe[model_id][joint_id]

                scene_mpjpe[im_filename][model_id]["overall"] = np.mean(frame_mpjpe[model_id], axis=0) # Model overall
                scene_pa_mpjpe[im_filename][model_id]["overall"] = np.mean(frame_pa_mpjpe[model_id], axis=0)

            scene_mpjpe[im_filename]["overall"] = np.mean(frame_mpjpe) # frame overall
            scene_pa_mpjpe[im_filename]["overall"] = np.mean(frame_pa_mpjpe)

            scene_mpjpe[im_filename]["n_preds"] = len(frame_mpjpe)
            scene_pa_mpjpe[im_filename]["n_preds"] = len(frame_pa_mpjpe)
            
            # Visual results
            if save_results:
                if outputs == None:
                    cv2.imwrite("{}/{}/{}".format(results_dir, scene_name, im_filename), empty_image)
                else:
                    cv2.imwrite("{}/{}/{}".format(results_dir, scene_name, im_filename), outputs["rendered_image"])

        # record scene error
        mpjpe[scene_name] = scene_mpjpe
        pa_mpjpe[scene_name] = scene_pa_mpjpe

        # save results as json at each scene
        with open("{}mpjpe_{}.json".format(results_dir, experiment_name), "w+") as filepointer:
            json.dump(mpjpe, filepointer)

        with open("{}pa_mpjpe_{}.json".format(results_dir, experiment_name), "w+") as filepointer:
            json.dump(pa_mpjpe, filepointer)


    print("Took {}mins".format((time.time() - start)/60))
