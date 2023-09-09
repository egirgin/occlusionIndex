import os
import numpy as np

import bev, cv2, json

all_dataset = "selected_list.txt"

subdatasets = os.listdir("./subdatasets")
"""
print(outputs.keys())
print(outputs["cam"])
print(outputs["global_orient"])
print(outputs["smpl_betas"])
print(outputs["smpl_thetas"])
print(outputs["cam_trans"])

"""


def show_img(img, window_name):
  cv2.imshow(str(window_name), img)

  cv2.waitKey(0)

  cv2.destroyAllWindows()

def match_pred_gt(preds, gt_smpl): # all pred vs one gt

  error_dist_list = []
  for pred_smpl_id in range(len(preds["smpl_thetas"])):

    temp_error = mpjpe(preds["smpl_thetas"][pred_smpl_id].reshape(24, 3), gt_smpl["parm_pose"])

    error_dist_list.append(temp_error)

  error_dist_list = np.array(error_dist_list)
  smallest_error_id = error_dist_list.argmin()

  smallest_error = error_dist_list[smallest_error_id]

  return smallest_error_id, smallest_error


def mpjpe(pred, gt):
  return np.sum(np.sqrt((pred - gt) ** 2))

def get_list(path):
  with open(path, "r") as img_list:
    input_list = img_list.read().splitlines()

  return input_list


def run_bev_once(img_path, dataset_name):
  os.makedirs("bev_results", exist_ok=True)
  os.makedirs("bev_results/{}".format(dataset_name[:-4]), exist_ok=True)

  smpl_path = "./gtSMPL/" + img_path.split("/")[-1][:-3] + "txt"

  with open(smpl_path, "r") as smpl_gt_file:
    smpl_gt = smpl_gt_file.read().splitlines()

  outputs = bev_model(cv2.imread(img_path))  # please note that we take the input image in BGR format (cv2.imread).

  per_frame_error = []

  for i in range(len(smpl_gt)):
    smpl_dict = json.loads(smpl_gt[i])

    error_id, pose_error = match_pred_gt(preds=outputs, gt_smpl=smpl_dict)

    shape_error = mpjpe(outputs["smpl_betas"][error_id][:-1], smpl_dict["parm_shape"])


    #individual_error = mpjpe(outputs["smpl_thetas"][closest_id].reshape(24,3), smpl_dict["parm_pose"])
    #individual_error = mpjpe(outputs["smpl_betas"], smpl_dict["parm_shape"])
    per_frame_error.append(pose_error + shape_error)


  cv2.imwrite("./bev_results/{}/{}.png".format(dataset_name[:-4], img_path.split("/")[-1][:-4]), outputs["rendered_image"])

  num_preds = len(outputs["smpl_thetas"])

  num_gt = len(smpl_gt)

  non_found = np.abs(num_gt - num_preds)

  frame_error = np.mean(per_frame_error)

  frame_error += (frame_error * non_found)

  #result_img = cv2.imread("processed_imgs/{}".format(img_path.split("/")[-1]))

  #show_img(result_img, str(frame_error) + " " +dataset_name[:-4])


  return frame_error

if __name__ == '__main__':
  settings = bev.main.default_settings
  # settings is just a argparse Namespace. To change it, for instance, you can change mode via
  # settings.mode='video'
  settings.render_mesh = True
  settings.show = False
  settings.show_items = "mesh,mesh_bird_view"
  #settings.save_path = "."
  #settings.save_video = True
  bev_model = bev.BEV(settings)

  print(settings)

  errors = []

  for sublist in subdatasets:
    input_list = get_list("./subdatasets/{}".format(sublist))

    sublist_error = []

    for img_path in input_list:
      frame_error = run_bev_once(img_path, dataset_name=sublist)

      sublist_error.append(frame_error)


    errors.append([sublist, np.mean(sublist_error)])

  # NOW ONLY METRICS
  ks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 500]

  error_results = []

  for k in ks:
    values = []
    occ_error = 0
    crowd_error = 0
    for sub_error in errors:
      if str(k)+".txt" in sub_error[0]:
        if "occlusion" in sub_error[0]:
          occ_error = sub_error[1]
        elif "crowd" in sub_error[0]:
          crowd_error = sub_error[1]


    diff = ((occ_error - crowd_error) / min([occ_error, crowd_error])) * 100

    values = [k, crowd_error, occ_error, diff]
    error_results.append(values)

  print(error_results)