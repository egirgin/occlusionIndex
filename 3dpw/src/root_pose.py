import os, sys, pickle, time, math, argparse
from visualize import *
from utils import *
from sklearn.cluster import KMeans
from smpl_np_romp import SMPLModel

if __name__ == '__main__':

    dataset_config = {
        "img_folder_path": "/home/tuba/Documents/emre/thesis/dataset/3dpw/imageFiles/",
        "mask_folder_path": "/home/tuba/Documents/emre/thesis/occlusionIndex/3dpw/masks/",
        "annotation_folder_path": "/home/tuba/Documents/emre/thesis/dataset/3dpw/sequenceFiles/sequenceFiles/",
    }
    scene_list = os.listdir(dataset_config["img_folder_path"])
    """
    scene_list = ['courtyard_warmWelcome_00', 'courtyard_captureSelfies_00', 
              'courtyard_dancing_01', 'courtyard_goodNews_00',
                'courtyard_giveDirections_00', 'courtyard_hug_00', 
                'courtyard_dancing_00', 'courtyard_basketball_00', 
              'courtyard_shakeHands_00', 'downtown_bar_00']
    
    other_scene_lists = [ # either no occlusion or neglectable occlusion 
        'downtown_cafe_00', 'downtown_arguing_00', 'downtown_bus_00', 'courtyard_rangeOfMotions_01',
        'courtyard_capoeira_00', 'courtyard_rangeOfMotions_00', 'courtyard_drinking_00', 'courtyard_arguing_00'
    ]
    """
    
    ################################################################################################################
    parser = argparse.ArgumentParser(description="3dpw arg parser")

    # Check the paths below for different configs
    parser.add_argument('-s', '--scene', choices=scene_list + ['all'],  default="courtyard_basketball_00", help='Scene of 3DPW')

    args = parser.parse_args()
    ################################################################################################################
    
    draw = False

    root_poses = []
    img_paths = []
    model_ids = []
    target_theta = None
    """
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
            #sys.exit()
            continue

        duration = []
        
        for frame_id in range(scene_length):
            start_time = time.time()

            image_filename = "image_{}.jpg".format(str(frame_id).zfill(5))

            remaining_secs = np.mean(duration)*(scene_length-frame_id)
            print("%{:.2f} Processing {}'s {}... ETA: {:.0f}mins {:.0f}secs".format(frame_id*100/scene_length, scene_name, image_filename, remaining_secs//60, remaining_secs%60))
            
            num_models = len(seq["betas"])

            thetas = np.zeros((num_models, 24, 3))
            #betas = np.zeros((num_models, 10))

            for model_id in range(num_models):

                #model_betas = seq["betas"][model_id]

                model_thetas = seq["poses"][model_id][frame_id].reshape(24,3)
                if scene_name == "courtyard_box_00" and frame_id == 152:
                    target_theta = model_thetas
                    print(target_theta)
                    sys.exit()
                
                root_poses.append(model_thetas[0])
                img_paths.append("{}/{}".format(scene_name, image_filename))
                model_ids.append(model_id)


            end_time = time.time()

            duration.append(end_time-start_time)

    root_img_pairs = [[root_poses[i], img_paths[i], model_ids[i]] for i in range(len(root_poses))]

    root_poses = np.array(root_poses)
    root_poses = root_poses[:, 1].reshape(-1,1)
    print(root_poses.shape)


       
    # Create a KMeans instance with 4 clusters
    kmeans = KMeans(n_clusters=7)

    # Fit the data to the KMeans model
    kmeans.fit(root_poses)

    # Get the cluster centers
    root_clusters = kmeans.cluster_centers_
       
    root_clusters = np.sort(root_clusters.reshape(-1))

    # Cluster centers will be a 2D array with 4 rows (4 centers) and 3 columns (x, y, z coordinates)
    print("K-Means Cluster Centers:")
    print(root_clusters)
    
    """
    target_theta = [
        [-3.91611719e-01, -2.47961513e-01, -1.06975715e-01],
        [ 1.31079257e-01,  1.86173400e-02,  1.77975086e-01],
        [ 3.31568905e-01, -5.54897275e-02, -1.02950522e-01],
        [-1.71818306e-01,  2.66406996e-02,  5.26088930e-02],
        [ 7.58962011e-02,  8.08460330e-02,  1.15416994e-02],
        [ 2.85195671e-01, -5.36367868e-02, -4.08388754e-02],
        [ 6.02581366e-01,  2.35010770e-02,  5.09446289e-02],
        [ 1.07246845e-01,  9.65148394e-02, -3.53570361e-02],
        [-7.33138435e-02, -1.75488457e-01,  2.25790442e-01],
        [ 2.37125514e-01,  2.25737006e-02,  7.69187970e-04],
        [-5.52904894e-03,  4.57220587e-02,  1.28314025e-01],
        [-9.64286265e-02,  3.60184229e-01, -4.50770809e-01],
        [ 1.14150404e-01,  1.75168057e-01, -1.37797621e-01],
        [-1.37432533e-01, -2.61468415e-01, -6.78954583e-02],
        [-8.27229677e-02,  3.14764093e-01,  1.72046653e-02],
        [ 2.90030113e-01,  1.01098268e-01, -1.58259129e-01],
        [-1.02505283e-01, -1.99803359e-01, -1.21530821e+00],
        [-1.92552641e-01,  6.15622486e-02,  9.91252982e-01],
        [-1.97108116e-02, -1.16693258e+00,  2.86424241e-01],
        [-3.86780972e-01,  1.24169639e+00, -1.01591614e-01],
        [ 1.91640667e-01,  4.99772639e-02,  5.35996402e-01],
        [ 1.93943760e-01, -1.12992109e-01, -3.51523125e-01],
        [-2.27476899e-01, -5.48938299e-02, -2.39892536e-01],
        [-1.60815852e-01,  1.04363734e-01,  2.73056604e-01],
    ]


    root_clusters = [-2.55617043, -1.74412909, -0.90232659, -0.0517622,   0.77182611,  1.57658457, 2.55470732] 
    
    prev_objs = os.listdir("./3dpw/src/")
    for i in range(len(prev_objs)):
        if ".obj" in prev_objs[i]:
            os.remove("./3dpw/src/" + prev_objs[i])

    smpl_m = SMPLModel('/home/tuba/Documents/emre/thesis/models/converted/SMPL_MALE.pkl')
    thetas = np.zeros((24, 3)) # (np.random.rand(24,3) - 1) * 0.6 #np.zeros((24, 3))
    thetas[13:] = target_theta[13:]
    trans = np.zeros(3)
    for center_idx, center in enumerate(root_clusters):
        thetas[0] = [0, 0, 0]
        thetas[0][1] = center
        degrees = np.rad2deg(center)
        #if degrees < 0:
        #    degrees = 360 + degrees 
        print(int(degrees))
        trans[0] = (center_idx*1)
        trans[1] = 0

        smpl_m.set_params(beta=np.zeros(10), pose=thetas, trans=trans)

        smpl_m.save_to_obj('./3dpw/src/{}.obj'.format(center_idx))

    
    """
    -146
    -99
    -51
    -2
    44
    90
    146
    """

    """
    img_clusters = [[]] * len(root_clusters)
    model_clusters = [[]] * len(root_clusters)

    results = []

    for idx, (root_pose, img_path, model_id) in enumerate(root_img_pairs):
        
        # Calculate the Euclidean distances between the new point and all points in the list
        distances = np.abs(root_clusters - root_pose[1])
        #print(distances)

        # Find the index of the closest point
        new_cluster = np.argmin(distances)

        results.append([new_cluster, img_path, model_id])

        img_clusters[new_cluster].append(img_path)      
        model_clusters[new_cluster].append(model_id)      
    
    result_path = "./3dpw/selected_frames/self_occlusion_list.txt"
    for cls in img_clusters:

        with open(result_path, "w+") as dump_file :
            for result in results:
                dump_file.write(
                    "{} {} {}\n".format(result[0], result[1], result[2])
                )

    """