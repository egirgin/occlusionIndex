import os, sys, argparse, time
import numpy as np
import romp, bev, cv2, json
import pandas as pd
from scipy.spatial.transform import Rotation as R

from utils import annotation_3dpw, compute_error, coco2smpl, computer_error_simple, computer_error_simple_joint
from error_metric import compute_similarity_transform

# path check is a must

global dataset, dataset_path, sublist, imglist_file, results_dir, save_results, model_type

scene_list = ['courtyard_dancing_01','courtyard_hug_00', 
              'courtyard_warmWelcome_00','courtyard_captureSelfies_00', 
              'courtyard_goodNews_00', 'courtyard_giveDirections_00',
                'courtyard_dancing_00', 'courtyard_basketball_00', 
              'courtyard_shakeHands_00', 'downtown_bar_00']

scene_list = ["outdoors_freestyle_00"]

parser = argparse.ArgumentParser(description="3dpw arg parser")

# Check the paths below for different configs
parser.add_argument('-m', '--model', choices=["romp", "bev"], default="romp", help='Model to be used.')
parser.add_argument('-d', '--save_results', action='store_true', default=True, help='Save results of model.')

args = parser.parse_args()
experiment_name = "self_occlusion"

################################################################################################################
model_type = args.model

dataset = "3dpw" 

dataset_parent_folder_path = "/home/tuba/Documents/emre/thesis/dataset/3dpw/"

save_results = True # args.save_results

################################################################################################################
def draw_keypoints(image, keypoints, kp_color=(255, 255, 255)):
    num_models = len(keypoints)

    for model_id in range(num_models):
        for kp_id, kp in enumerate(keypoints[model_id]):
            image = cv2.circle(image, (int(kp[0]), int(kp[1])), 1, kp_color, 2)
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

    mpjpe_keypoint = {}
    pa_mpjpe_keypoint = {}

    mpjpe_regional = {}
    pa_mpjpe_regional = {}
    ####################RUN########################################################################
    scene_list = ["outdoors_slalom_00"] 

    for scene_idx, scene_name in enumerate(scene_list):
        os.makedirs(results_dir + "/" + scene_name, exist_ok=True)
        print("Testing {} on {}'s {} scene...".format(model_type, dataset, scene_name))

        dataset_imgs_path = dataset_parent_folder_path + "imageFiles/{}/".format(scene_name)
        dataset_annotations_path = dataset_parent_folder_path + "sequenceFiles/sequenceFiles/"

        scene_img_list = os.listdir(dataset_imgs_path)

        ############### ERROR PER SCENE###########################################################
        scene_mpjpe = {}
        scene_pa_mpjpe = {}   

        scene_mpjpe_keypoint = {}
        scene_pa_mpjpe_keypoint = {}

        scene_mpjpe_regional = {}
        scene_pa_mpjpe_regional = {}

        #######################################RUN OVER SAMPLES####################################
        #scene_img_list = ["image_00171.jpg"] 
        """
        courtyard_dancing_00/image_00349.jpg
        courtyard_dancing_00/image_00181.jpg
        ourtyard_dancing_00/image_00180.jpg
        courtyard_dancing_00/image_00347.jpg
        courtyard_dancing_00/image_00182.jpg
        downtown_bar_00/image_00741.jpg
        downtown_bar_00/image_01328.jpg
        downtown_bar_00/image_00737.jpg
        downtown_bar_00/image_00739.jpg
        downtown_bar_00/image_00738.jpg
        downtown_bar_00/image_01333.jpg
        downtown_bar_00/image_00748.jpg
        downtown_bar_00/image_00752.jpg
        """
        
        for i, im_filename in enumerate(scene_img_list): # first two are index and subset order
            print("Scene: {} | Progress: {:.2f}".format(scene_name, i*100/len(scene_img_list)))

            # get GTs
            thetas_gt, betas_gt, keypoints2d_gt, camera_parameters = annotation_3dpw(annotation_filepath=dataset_annotations_path, scene_name=scene_name, im_filename =im_filename)
            num_models = thetas_gt.shape[0]

            # predict
            img = cv2.imread(dataset_imgs_path + im_filename)
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

            #img = draw_keypoints(image=img, keypoints=keypoints2d_gt[0].reshape(-1,24,2), kp_color=(0,0,255))
            #img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))
            #cv2.imshow(im_filename, img)

            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #sys.exit()

            # Calculate error
            frame_mpjpe, frame_pa_mpjpe = computer_error_simple_joint(thetas_gt=thetas_gt, thetas_pred=thetas_pred, keypoints2d_gt=keypoints2d_gt, keypoints2d_pred=keypoints2d_pred)
            #print(frame_mpjpe)

            #render_result = cv2.resize(outputs["rendered_image"], (int(outputs["rendered_image"].shape[1] * 0.25), int(outputs["rendered_image"].shape[0] * 0.25)))
            #cv2.imshow(im_filename, render_result)

            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #sys.exit()

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

            print(frame_pa_mpjpe)
            print(np.mean(frame_pa_mpjpe[0]))
            print(np.mean(frame_pa_mpjpe[0][:16]))

            # Using cv2.imshow() method 
            # Displaying the image 
            cv2.imshow("test", outputs["rendered_image"]) 
            
            # waits for user to press any key 
            # (this is necessary to avoid Python kernel form crashing) 
            cv2.waitKey(0) 
            
            # closing all open windows 
            cv2.destroyAllWindows() 


            sys.exit()

        # record scene error
        mpjpe[scene_name] = scene_mpjpe
        pa_mpjpe[scene_name] = scene_pa_mpjpe

        # save results as json at each scene
        with open("{}mpjpe_{}.json".format(results_dir, experiment_name), "w+") as filepointer:
            json.dump(mpjpe, filepointer)

        with open("{}pa_mpjpe_{}.json".format(results_dir, experiment_name), "w+") as filepointer:
            json.dump(pa_mpjpe, filepointer)


    print("Took {}mins".format((time.time() - start)/60))
