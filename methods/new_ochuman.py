import os, sys, argparse, time
import numpy as np
import romp, bev, cv2, json
from scipy.spatial.transform import Rotation as R

from utils import computer_error_simple_joint, computer_error_simple_joint_ochuman
from ochuman_utils import get_annos, ochuman_annotation, get_smpl, coco2smpl, smpl2coco

from utils import ochuman_annotation_smpl, compute_error

# path check is a must

global dataset, dataset_path, sublist, imglist_file, results_dir, save_results, model_type

parser = argparse.ArgumentParser(description="ochuman arg parser")

subset_list = ["empty", "head", "torso", "left_lower", "left_upper", "right_lower", "right_upper", "right_body", "left_body", "upper_body", "lower_body"]

# Check the paths below for different configs
parser.add_argument('-m', '--model', choices=["romp", "bev"], default="romp", help='Model to be used.')
parser.add_argument('-d', '--save_results', action='store_true', default=False, help='Save results of model.')

args = parser.parse_args()
experiment_name = "joints"

################################################################################################################
model_type = args.model

dataset = "ochuman" 
dataset_parent_folder_path = "/home/tuba/Documents/emre/thesis/dataset/ochuman/OCHuman-20230218T083152Z-001/OCHuman/"
dataset_imgs_path = dataset_parent_folder_path + "images/"
dataset_smpl_path = "./ochuman/gtSMPL/"
dataset_annotations_path = dataset_parent_folder_path + "ochuman.json"

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

def draw_keypoints(image, keypoints):
    for kp_id, kp in enumerate(keypoints):
        image = cv2.circle(image, (int(kp[0]), int(kp[1])), 3, (0, 0, 255), 4)
    return image

if __name__ == '__main__':
    print("Testing {} on {}...".format(model_type, dataset))
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

    
    ###############################ERRORS#######################################################
    results_dir = "./methods/{}_results/{}/".format(model_type, dataset)
    os.makedirs(results_dir, exist_ok=True)

    mpjpe = {}
    pa_mpjpe = {}
    ####################RUN########################################################################
    annotation = get_annos(dataset_annotations_path)

    num_frames = len(annotation["images"])

    
    for frame_id in range(num_frames):
        print("Scene: {} | Progress: {:.2f}".format(dataset, frame_id*100/num_frames))

        # get GTs
        im_filename, _, img_keypoints_orig = ochuman_annotation(annotation, frame_id=frame_id)
        num_smpl, smpls = get_smpl(im_filename)

        if im_filename == None: # if there is missing mask or pose of a model
            continue

        if num_smpl == None:
            print("Frame {} No SMPL data, skipping...".format(frame_id))
            continue

        if len(img_keypoints_orig) != num_smpl:
            print("Frame {} SMPL does not match keypoints... Skipping...".format(frame_id))
            continue

        thetas_gt, betas_gt = ochuman_annotation_smpl(im_filename=dataset_smpl_path + im_filename)
        num_models = len(img_keypoints_orig)


        img_keypoints = img_keypoints_orig[:, :14, :2]


        # predict
        img = cv2.imread(dataset_imgs_path +im_filename)
        img_height, img_width = img.shape[:2]
        outputs = model(img)  # please note that we take the input image in BGR format (cv2.imread).
        if outputs == None: # If could not find any human
            empty_image = np.zeros_like(img)
            thetas_pred = np.zeros((1,24,3))
            betas_pred = np.zeros((1,10))
            keypoints2d_pred = np.zeros(1,14,2)
        else:
            thetas_pred = outputs["smpl_thetas"].reshape(-1, 24, 3)
            betas_pred = outputs["smpl_betas"][:, :10].reshape(-1, 10)
            keypoints2d_pred = outputs["pj2d_org"][:, :24, :]
            keypoints2d_pred = smpl2coco(keypoints2d_pred)

        for model_id in range(num_models):
            for kp_id in range(14):
                if img_keypoints[model_id][kp_id][0] == 0:
                    keypoints2d_pred[model_id][kp_id] = [0,0]

            img_keypoints[model_id][12] = img_keypoints_orig[model_id][16][:-1]

        frame_mpjpe, frame_pa_mpjpe, matched_pairs = computer_error_simple_joint_ochuman(thetas_gt=thetas_gt, thetas_pred=thetas_pred, keypoints2d_gt=img_keypoints, keypoints2d_pred=keypoints2d_pred)

        """
        # matched_pairs: gt, pred
        if "4852" in im_filename: 

            for i in range(matched_pairs.shape[0]):
                [gt_id, pred_id] = matched_pairs[i]

                gt_img = draw_keypoints(img.copy(), img_keypoints[gt_id])
                pred_img = draw_keypoints(img.copy(), keypoints2d_pred[pred_id])

                concat_img = np.concatenate((gt_img, pred_img, outputs["rendered_image"][:,img_width:2*img_width]), axis=1)

                cv2.imshow("test", concat_img)

                cv2.waitKey(0)

                cv2.destroyAllWindows()

            sys.exit()

        """
        


        # Record frame error
        mpjpe[im_filename] = {}
        pa_mpjpe[im_filename] = {}
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
            mpjpe[im_filename][model_id] = {
                "overall": 0.0
            }
            pa_mpjpe[im_filename][model_id] = {
                "overall": 0.0
            }
            for joint_id in range(24):
                if frame_mpjpe[model_id][joint_id] == 9999:
                    #print("Could not find the human at {} {}!".format(scene_name, im_filename))
                    pass

                mpjpe[im_filename][model_id][joint_id] = frame_mpjpe[model_id][joint_id] 
                pa_mpjpe[im_filename][model_id][joint_id] = frame_pa_mpjpe[model_id][joint_id]

            mpjpe[im_filename][model_id]["overall"] = np.mean(frame_mpjpe[model_id], axis=0) # Model overall
            pa_mpjpe[im_filename][model_id]["overall"] = np.mean(frame_pa_mpjpe[model_id], axis=0)

        mpjpe[im_filename]["overall"] = np.mean(frame_mpjpe) # frame overall
        pa_mpjpe[im_filename]["overall"] = np.mean(frame_pa_mpjpe)

        mpjpe[im_filename]["n_preds"] = len(frame_mpjpe)
        pa_mpjpe[im_filename]["n_preds"] = len(frame_pa_mpjpe)

        # Visual results
        if save_results:
            if outputs == None:
                cv2.imwrite("{}/{}".format(results_dir, im_filename), empty_image)
            else:
                cv2.imwrite("{}/{}".format(results_dir, im_filename), outputs["rendered_image"])

    # save results as json at each scene
    with open("{}mpjpe_{}.json".format(results_dir, experiment_name), "w+") as filepointer:
        json.dump(mpjpe, filepointer)

    with open("{}pa_mpjpe_{}.json".format(results_dir, experiment_name), "w+") as filepointer:
        json.dump(pa_mpjpe, filepointer)