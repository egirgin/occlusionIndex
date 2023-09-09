import os, sys, argparse
import numpy as np
import romp, bev, cv2, json

from utils import ochuman_annotation, compute_error, coco2smpl

# path check is a must

global dataset, dataset_path, sublist, imglist_file, results_dir, save_results, model_type

parser = argparse.ArgumentParser(description="ochuman arg parser")

# Check the paths below for different configs
parser.add_argument('-m', '--model', choices=["romp", "bev"], default="romp", help='Model to be used.')
parser.add_argument('-s', '--subset', choices=["empty, head, torso, left_lower, left_upper, right_lower, right_upper"], default="empty", help='Subset to be used')
parser.add_argument('-e', '--error_modified', action='store_true', default=True, help='Use modified MPJPE')
parser.add_argument('-d', '--save_results', action='store_true', default=True, help='Save results of model.')

args = parser.parse_args()

################################################################################################################
model_type = args.model

dataset = "ochuman" 
dataset_parent_folder_path = "/home/tuba/Documents/emre/thesis/dataset/ochuman/OCHuman-20230218T083152Z-001/OCHuman/"
dataset_imgs_path = dataset_parent_folder_path + "images/"
dataset_smpl_path = dataset_parent_folder_path + "gtSMPL/"

sublist = "{}_subset.txt".format(args.subset)
imglist_file = "./methods/selected_frames/{}/{}".format(dataset, sublist)

modified_mpjpe = args.error_modified

save_results = args.save_results
if save_results:
    results_dir = "./methods/{}_results/{}/{}".format(model_type, dataset, sublist[:-4])
    os.makedirs(results_dir, exist_ok=True)
################################################################################################################
def get_img_list(selected_list_path):
    with open(selected_list_path, "r") as img_list:
        input_list = img_list.read().splitlines()

    filelist = []
    for frame in input_list:
        filelist.append(frame.split()[0])

    return filelist

if __name__ == '__main__':
    if model_type == "romp":
        settings = romp.main.default_settings
        # settings is just a argparse Namespace. To change it, for instance, you can change mode via
        # settings.mode='video'
        settings.render_mesh = True
        settings.show = False
        settings.show_items = "mesh,mesh_bird_view"
        #settings.save_path = "."
        #settings.save_video = True
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
        model = bev.BEV(settings)
        print(bev.main.default_settings)

    
    errors = []
    pa_errors = []

    input_list = get_img_list(imglist_file)

    for im_filename in input_list:

        # get GTs
        thetas_gt, betas_gt = ochuman_annotation(im_filename=dataset_smpl_path + im_filename)
        
        if modified_mpjpe:
            # TODO read from annotations
            num_ppl = thetas_gt.shape[0]
            occlusion_mask = np.random.choice([True, False], size=(num_ppl, 19))
            occlusion_mask = coco2smpl(occlusion_mask)
        else:
            num_ppl = thetas_gt.shape[0]
            occlusion_mask = np.random.choice([True], size=(num_ppl, 19))
            occlusion_mask = coco2smpl(occlusion_mask)

        # predict
        outputs = model(cv2.imread(dataset_imgs_path +im_filename))  # please note that we take the input image in BGR format (cv2.imread).
        thetas_pred = outputs["smpl_thetas"].reshape(-1, 24, 3)
        betas_pred = outputs["smpl_betas"][:, :10].reshape(-1, 10)
        
        mpjpe, pa_mpjpe = compute_error(thetas_pred=thetas_pred, thetas_gt=thetas_gt, 
                                        betas_pred=betas_pred, betas_gt=betas_gt, 
                                        occlusion_masks=occlusion_mask, pose_coeff=0.9, shape_coeff=0.1)
        
        errors.append(mpjpe)
        pa_errors.append(pa_mpjpe)

        if save_results:
            cv2.imwrite("{}/{}".format(results_dir, im_filename), outputs["rendered_image"])
        break # TODO
    
    with open(results_dir+ "/results.txt", "w+") as result_file:
        mpjpe_s = "MPJPE for {} {}: {} \n".format(dataset, sublist[:-4], np.mean(errors))
        pa_mpjpe_s = "PA-MPJPE for {} {}: {} \n".format(dataset, sublist[:-4], np.mean(pa_errors))
        
        result_file.write(mpjpe_s)
        result_file.write(pa_mpjpe_s)
        
        print(mpjpe_s[:-1])
        print(pa_mpjpe_s[:-1])
