import os, sys
import romp, cv2, json, pickle
import numpy as np

from error_metric import match_models, sort_by_order, compute_mpjpe, compute_shape_error, match_neutral, compute_mpjpe_original, compute_mpjpe_per_joint
from agora_projection import project2d


def coco2smpl(coco_pose):
    """
    smpl_format = ["pelvis", "left_hip", "right_hip", "lower_spine", "left_knee", "right_knee", # 0-5
        "middle_spine", "left_ankle", "right_ankle", "upper_spine", "left_foot", "right_foot", # 6-11
        "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder", # 12-17
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand"] # 18-23

    coco_format =  ['right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder',
        'left_elbow', 'left_wrist', 'right_hip', 'right_knee', 'right_ankle',
        'left_hip', 'left_knee', 'left_ankle', 'head', 'neck', 'right_ear',
        'left_ear', 'nose', 'right_eye', 'left_eye']
    """

    num_models = len(coco_pose)

    smpl_poses = np.zeros((num_models, 24), dtype=bool)

    common_joints = [(1, 9), (2, 6), (4, 10), (5, 7), (7, 11), (8, 8), (12, 13), (15, 12), (16, 3), (17, 0), (18, 4), (19, 1), (20, 5), (21, 2)]

    for model_id in range(num_models):

        for (smpl_joint, coco_joint) in common_joints:
            smpl_poses[model_id][smpl_joint] = coco_pose[model_id][coco_joint]

        smpl_poses[model_id][0] = smpl_poses[model_id][1] and smpl_poses[model_id][2] # pelvis = left_hip and right_hip
        smpl_poses[model_id][3] = smpl_poses[model_id][0] # lower_spine = pelvis
        smpl_poses[model_id][9] = smpl_poses[model_id][16] and smpl_poses[model_id][17] # upper_spine = left_shoulder and right_shoulder
        smpl_poses[model_id][6] = smpl_poses[model_id][3] and smpl_poses[model_id][9] # middle_spine = lower_spine and upper_spine
        smpl_poses[model_id][10] = smpl_poses[model_id][7] # left_foot = left_ankle
        smpl_poses[model_id][11] = smpl_poses[model_id][8] # right_foot = right_ankle
        smpl_poses[model_id][13] = smpl_poses[model_id][16] and smpl_poses[model_id][12] # left_collar = left_shoulder and neck
        smpl_poses[model_id][14] = smpl_poses[model_id][17] and smpl_poses[model_id][12] # right_collar = right_shoulder and neck
        smpl_poses[model_id][22] = smpl_poses[model_id][20] # left_hand = left_wrist
        smpl_poses[model_id][23] = smpl_poses[model_id][21] # right_hand = right_wrist

    return smpl_poses

def computer_error_simple(thetas_pred, thetas_gt):
    
    thetas_gt, thetas_pred, matched_pairs = match_neutral(fixed=thetas_gt, changed=thetas_pred, match_by="root")
    return compute_mpjpe_original(preds3d=thetas_pred, gt3ds=thetas_gt)

def computer_error_simple_joint(thetas_pred, thetas_gt, keypoints2d_gt, keypoints2d_pred):
    
    thetas_gt, thetas_pred, matched_pairs = match_neutral(fixed=thetas_gt, changed=thetas_pred, keypoints2d_gt=keypoints2d_gt, keypoints2d_pred=keypoints2d_pred, match_by="keypoints2d")
    
    #print(matched_pairs)

    return compute_mpjpe_per_joint(preds3d=thetas_pred, gt3ds=thetas_gt)

def computer_error_simple_joint_ochuman(thetas_pred, thetas_gt, keypoints2d_gt, keypoints2d_pred):
    
    thetas_gt, thetas_pred, matched_pairs = match_neutral(fixed=thetas_gt, changed=thetas_pred, keypoints2d_gt=keypoints2d_gt, keypoints2d_pred=keypoints2d_pred, match_by="keypoints2d")
    
    #print(matched_pairs)

    frame_mpjpe, frame_pa_mpjpe = compute_mpjpe_per_joint(preds3d=thetas_pred, gt3ds=thetas_gt)

    return frame_mpjpe, frame_pa_mpjpe, matched_pairs


def compute_error(thetas_pred, thetas_gt, betas_pred, betas_gt, occlusion_masks, pose_coeff=0.9, shape_coeff=0.1):

    thetas_pred, thetas_gt, matched_pairs = match_models(preds=thetas_pred, gts=thetas_gt, match_by="root")

    betas_pred = sort_by_order(array=betas_pred, order=matched_pairs[:,0])
    betas_gt = sort_by_order(array=betas_gt, order=matched_pairs[:,1])
    occlusion_masks = sort_by_order(array=occlusion_masks, order=matched_pairs[:, 1])
    
    occlusion_masks = np.invert(occlusion_masks.astype(bool))

    mpjpe, pa_mpjpe, modified_mpjpe, modified_pa_mpjpe = compute_mpjpe(thetas_pred, thetas_gt, mask=occlusion_masks)

    shape_error = compute_shape_error(betas_pred, betas_gt)

    total_error = pose_coeff*mpjpe + shape_coeff*shape_error

    total_error_pa = pose_coeff*pa_mpjpe + shape_coeff*shape_error

    ###################### MODIFIED MPJPE#############################333

    total_error_modified = pose_coeff*modified_mpjpe + shape_coeff*shape_error

    total_error_pa_modified = pose_coeff*modified_pa_mpjpe + shape_coeff*shape_error

    return total_error, total_error_pa, total_error_modified, total_error_pa_modified

def ochuman_annotation_smpl(im_filename):

    smpl_path = im_filename[:-3] + "txt"

    with open(smpl_path, "r") as smpl_gt_file:
        smpl_gt = smpl_gt_file.read().splitlines()
    
    num_gt = len(smpl_gt)

    thetas = np.zeros((num_gt, 24, 3))
    betas = np.zeros((num_gt, 10))

    for i in range(len(smpl_gt)):
        smpl_dict = json.loads(smpl_gt[i])
        thetas[i] = smpl_dict["parm_pose"]
        betas[i] = smpl_dict["parm_shape"]

    return thetas, betas

def agora_annotation(annotation_filepath, smpl_path):
    """
    record = {
            "smpl_path" :  N
            "smplx_path" : N
            "keypoints2d" :  # Nx45x2 # convert to COCO format explicitly
            "keypoints3d" :  # Nx45x3 
            "location" :  # Nx4 (X,Y,Z,Yaw) yaw in degrees
            "cam_extrinsics" :  # 1x4 (X,Y,Z,Yaw)
        }
    """
    with open(annotation_filepath[:-4] + ".json", "r") as annotation:
            anno = json.load(annotation)
    
    num_models = len(anno["smpl_path"])
    thetas = np.zeros((num_models, 24, 3))
    betas = np.zeros((num_models, 10))

    for model_id in range(num_models):
        model_smpl_path = smpl_path + anno["smpl_path"][model_id]
        model_smpl_path = model_smpl_path[:-3] + "pkl"

        with open(model_smpl_path, "rb") as smpl_file:
            smpl = pickle.load(smpl_file)
            # keys :['translation', 'root_pose', 'body_pose', 'betas', 'joints', 'faces', 'vertices', 'full_pose', 'v_shaped']

        model_betas = smpl["betas"].detach().cpu().numpy()[0]

        model_root_pose = smpl["root_pose"].detach().cpu().numpy()[0]
        model_body_pose = smpl["body_pose"].detach().cpu().numpy()[0]
        
        #joints = smpl["joints"].detach().cpu().numpy()[0] # difference between body pose ? check which one is smpl theta

        model_thetas = np.vstack((model_root_pose, model_body_pose))

        thetas[model_id] = model_thetas
        betas[model_id] = model_betas

    return thetas, betas

def annotation_agora(smpl_path, smpl_filenames, num_models, location, imgHeight, imgWidth, scene="hdri_50mm"):
    # these are parameters for hdri_50mm, for other scenes see: https://github.com/pixelite1201/agora_evaluation/blob/master/agora_evaluation/projection.py
    if scene == "hdri_50mm":
        ground_plane = [0, 0, 0]
        scene3d = False
        focalLength = 50
        camPosWorld = [0, 0, 170]
        camYaw = 0
        camPitch = 0
    elif scene == "archviz":
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [-490, 170, 265]
        camYaw = -45
        camPitch = 30

    dslr_sens_width = 36
    dslr_sens_height = 20.25

    thetas = np.zeros((num_models, 24, 3))
    betas = np.zeros((num_models, 10))
    keypoints2d = np.zeros((num_models, 24, 2))

    for model_id in range(num_models):
        model_smpl_path = smpl_path + smpl_filenames[model_id]
        model_smpl_path = model_smpl_path[:-3] + "pkl"

        with open(model_smpl_path, "rb") as smpl_file:
            smpl = pickle.load(smpl_file)
            # keys :['translation', 'root_pose', 'body_pose', 'betas', 'joints', 'faces', 'vertices', 'full_pose', 'v_shaped']

        model_betas = smpl["betas"].detach().cpu().numpy()[0]
        
        #joints = smpl["joints"].detach().cpu().numpy()[0] # difference between body pose ? check which one is smpl theta

        model_root_pose = smpl["root_pose"].detach().cpu().numpy()[0]
        model_body_pose = smpl["body_pose"].detach().cpu().numpy()[0]
        model_thetas = np.vstack((model_root_pose, model_body_pose))

        thetas[model_id] = model_thetas
        betas[model_id] = model_betas
        
        trans3d = location[model_id][:-1]

        gt2d, gt3d_camCoord = project2d(smpl["joints"].detach().cpu().numpy()[0], focalLength=focalLength, scene3d=scene3d,
                                    trans3d=trans3d,
                                    dslr_sens_width=dslr_sens_width,
                                    dslr_sens_height=dslr_sens_height,
                                    camPosWorld=camPosWorld,
                                    cy=imgHeight / 2,
                                    cx=imgWidth / 2,
                                    imgPath=None,
                                    yawSMPL=location[model_id][-1],
                                    ground_plane=ground_plane,
                                    debug_path="./agora/debug_agora_projection",
                                    debug=False,
                                    ind=-1,
                                    pNum=-1,
                                    meanPose=False, camPitch=camPitch, camYaw=camYaw)
        
        keypoints2d[model_id] = gt2d

    return thetas, betas, keypoints2d



def annotation_3dpw(annotation_filepath, scene_name, im_filename):

    def get_cam_params(seq, frame_id):

        intrinsic = seq["cam_intrinsics"]

        extrinsic = seq["cam_poses"][frame_id]

        R = extrinsic[:3,:3]
        t = extrinsic[:-1, -1]

        t = np.expand_dims(t, axis=1)

        return intrinsic, R, t, extrinsic
    
    def estimate_from_3d(seq, frame_id, model_id, keypoints3d):

        intrinsic, R, t, extrinsic = get_cam_params(seq, frame_id)

        estimated_keypoints_2d, _ = cv2.projectPoints(keypoints3d, R, t, intrinsic, None)

        estimated_keypoints_2d = np.squeeze(estimated_keypoints_2d, axis=1)

        return estimated_keypoints_2d


    if scene_name+".pkl" in os.listdir(annotation_filepath+ "train"):
        split = "train/"
    elif scene_name+".pkl" in os.listdir(annotation_filepath + "validation"):
        split = "validation/"
    elif scene_name+".pkl" in os.listdir(annotation_filepath + "test"):
        split = "test/"
    else:
        print("Sequence {} could not be found!".format(scene_name))
        sys.exit()

    annotation_filepath += split + scene_name

    seq_file_path = annotation_filepath + ".pkl"

    seq = pickle.load(open(seq_file_path, "rb"), encoding='latin1')


    img_id = int(im_filename.split("_")[-1][:-4])

    intrinsic, R, t, extrinsic = get_cam_params(seq, img_id)

    camera_matrices = {
        "intrinsics": intrinsic,
        "extrinsics": extrinsic,
        "rotation": R,
        "translation": t.reshape(-1)
    }

    num_models = len(seq["betas"])

    thetas = np.zeros((num_models, 24, 3))
    betas = np.zeros((num_models, 10))

    keypoints2d = np.zeros((num_models, 24, 2))

    for model_id in range(num_models):

        model_betas = seq["betas"][model_id]

        model_thetas = seq["poses"][model_id][img_id].reshape(24,3)

        joints3d = seq["jointPositions"][model_id][img_id].reshape(24,3)
        
        thetas[model_id] = model_thetas
        betas[model_id] = model_betas[:10]

        keypoints2d[model_id] = estimate_from_3d(seq=seq, frame_id=img_id, model_id=model_id, keypoints3d=joints3d)

    return thetas, betas, keypoints2d, camera_matrices
