from subset_creator_helper import *
from subset_weights import *

dataset = "agora" # 3dpw, agore


anno_root = "./{}/".format(dataset)
regional_occlusion_anno_path = anno_root + "roi_json"
occlusion_index_anno_path = anno_root + "occlusion_index"
crowd_index_anno_path = anno_root + "crowd_index"


final_error_algorithm = "standard" # standard, keypoint_visibility, truncation, region_visibility
##############################################################################################################################
category = "occlusion"  # or "self_occlusion"
learn_weights = False
##############################################################################################################################
method = "romp" # romp, bev
error_type ="pa_mpjpe_14_dec" # mpjpe_joints, pa_mpjpe_joints
error_path = "./methods/{}_results/{}/{}.json".format(method, dataset, error_type)

def replace_mean(array, merge_indices, new_index):

    new_value = np.mean(array[merge_indices])
    merged_array = np.delete(array, merge_indices)  # Remove 
    merged_array = np.insert(merged_array, new_index, new_value)  # Insert 
    
    return merged_array


def experiment1(): # per sample error with running meanplt.ylabel("Running Mean of {}".format(" ".join(error_type.split("_")[:-1]).upper()))

    weights = uniform_weights
    weights = list(weights.values())
    weights = normalize_to_sum_one(weights)

    occlusion_list_roi = process_occlusion_data(regional_occlusion_anno_path, category, weights, algorithm="regional_occlusion")
    occlusion_list_oi = process_occlusion_data(occlusion_index_anno_path, category, weights, algorithm="occlusion_index")
    occlusion_list_ci = process_occlusion_data(crowd_index_anno_path, category, weights, algorithm="crowd_index")
    for i in range(0, 100):
        #print(i+1)
        #print(occlusion_list_roi[i][1])
        #print("{:.3f}".format(occlusion_list_roi[i][2]))
        #print(occlusion_list_roi[i])

        path = ""

        for term in occlusion_list_roi[i][0].split("_"):
            path += term
            path += "\_"

        path = path[:-2]

        print("\item {}: {} Person ID: {}, ROI Score: {:.3f}".format(i+1, path, occlusion_list_roi[i][1], occlusion_list_roi[i][2]))
    sys.exit()

    k = min(len(occlusion_list_oi), len(occlusion_list_ci),  len(occlusion_list_roi))
    print(k)

    error_roi, occlusion_scores_roi, missing_error_roipredictions_roi = calculate_error(error_path=error_path, occlusion_path=anno_root, occlusion_list=occlusion_list_roi, k=k, error_algorithm=final_error_algorithm)
    error_oi, occlusion_scores_oi, missing_predictions_oi = calculate_error(error_path=error_path, occlusion_path=anno_root, occlusion_list=occlusion_list_oi, k=k, error_algorithm=final_error_algorithm)
    error_ci, occlusion_scores_ci, missing_predictions_ci = calculate_error(error_path=error_path, occlusion_path=anno_root, occlusion_list=occlusion_list_ci, k=k, error_algorithm=final_error_algorithm)

    error_roi = np.array(error_roi)
    error_oi = np.array(error_oi)
    error_ci = np.array(error_ci)

    error_roi = error_roi[:, 0]
    error_roi = running_mean(error_roi)

    error_oi = error_oi[:, 0]
    error_oi = running_mean(error_oi)

    error_ci = error_ci[:, 0]
    error_ci = running_mean(error_ci)

    x = np.arange(k)

    x = [i*100/k for i in x]

    plt.figure(figsize=(10, 8))  # Width, Height

    plt.plot(x, error_roi, label="Regional Occlusion Index")
    plt.plot(x, error_oi, label="Occlusion Index")
    plt.plot(x, error_ci, label="Crowd Index")

    plt.xlabel("Subset Size (%)")
    plt.ylabel("Running Mean of {}".format(" ".join(error_type.split("_")[:-2]).upper()))
    plt.legend(loc='best')  # You can specify the legend location
    #plt.title("ROMP's Performance on Top 10% Challenging Samples of 3DPW According to Different Indices")
    plt.tight_layout()

    plt.savefig("./agora/experiment_figures/experiment1_{}_{}.png".format(method, error_type))

    for j in [10, 20, 30, 40, 50, 60, 70]:

        idx = int((j*k)/100) 
        print("{:.2f}".format(error_roi[idx]))
        print("{:.2f}".format(error_oi[idx]))
        print("{:.2f}".format(error_ci[idx]))
        print("#############################")


    # Show the plot
    plt.show()
    plt.close()

def experiment2(): # number of missing predictions running sum

    weights = uniform_weights
    weights = list(weights.values())
    weights = normalize_to_sum_one(weights)

    occlusion_list_roi = process_occlusion_data(regional_occlusion_anno_path, category, weights, algorithm="regional_occlusion")
    occlusion_list_oi = process_occlusion_data(occlusion_index_anno_path, category, weights, algorithm="occlusion_index")
    occlusion_list_ci = process_occlusion_data(crowd_index_anno_path, category, weights, algorithm="crowd_index")
    
    k = len(occlusion_list_roi)
    error_roi, occlusion_scores_roi, missing_predictions_roi = calculate_error(error_path=error_path, occlusion_path=anno_root, occlusion_list=occlusion_list_roi, k=k, error_algorithm=final_error_algorithm)
    error_oi, occlusion_scores_oi, missing_predictions_oi = calculate_error(error_path=error_path, occlusion_path=anno_root, occlusion_list=occlusion_list_oi, k=k, error_algorithm=final_error_algorithm)
    error_ci, occlusion_scores_ci, missing_predictions_ci = calculate_error(error_path=error_path, occlusion_path=anno_root, occlusion_list=occlusion_list_ci, k=k, error_algorithm=final_error_algorithm)

    missing_predictions_roi = running_sum(missing_predictions_roi)

    missing_predictions_oi = running_sum(missing_predictions_oi)

    missing_predictions_ci = running_sum(missing_predictions_ci)

    xm = np.arange(len(missing_predictions_roi))

    xm = [i/len(occlusion_list_roi) for i in xm]

    plt.figure(figsize=(10, 8))  # Width, Height

    plt.plot(xm, missing_predictions_roi, label="regional_occlusion")
    plt.plot(xm, missing_predictions_oi, label="occlusion_index")
    plt.plot(xm, missing_predictions_ci, label="crowd_index")

    plt.xlabel("Subset Size (%)")
    plt.ylabel("Running Sum of # Missing Predictions")
    plt.legend(title="k", loc='best')  # You can specify the legend location
    plt.title("3DPW")
    plt.tight_layout()

    # Show the plot
    plt.show()
    plt.close()

def experiment3(): #MPJPE per occlusion
    weights = uniform_weights
    weights = list(weights.values())
    weights = normalize_to_sum_one(weights)

    occlusion_list_roi = process_occlusion_data(regional_occlusion_anno_path, category, weights, algorithm="regional_occlusion")
    occlusion_list_oi = process_occlusion_data(occlusion_index_anno_path, category, weights, algorithm="occlusion_index")
    occlusion_list_ci = process_occlusion_data(crowd_index_anno_path, category, weights, algorithm="crowd_index")
    
    k = min(len(occlusion_list_oi), len(occlusion_list_ci),  len(occlusion_list_roi))
    print(k)

    error_roi, occlusion_scores_roi, missing_predictions_roi = calculate_error(error_path=error_path, occlusion_path=anno_root, occlusion_list=occlusion_list_roi, k=k, error_algorithm=final_error_algorithm, drop_zero_occ=True)
    error_oi, occlusion_scores_oi, missing_predictions_oi = calculate_error(error_path=error_path, occlusion_path=anno_root, occlusion_list=occlusion_list_oi, k=k, error_algorithm=final_error_algorithm, drop_zero_occ=True)
    error_ci, occlusion_scores_ci, missing_predictions_ci = calculate_error(error_path=error_path, occlusion_path=anno_root, occlusion_list=occlusion_list_ci, k=k, error_algorithm=final_error_algorithm, drop_zero_occ=True)

    error_roi = np.array(error_roi)
    error_oi = np.array(error_oi)
    error_ci = np.array(error_ci)

    error_roi = error_roi[:, 0]
    error_roi = running_mean(error_roi)

    error_oi = error_oi[:, 0]
    error_oi = running_mean(error_oi)

    error_ci = error_ci[:, 0]
    error_ci = running_mean(error_ci)

    plt.figure(figsize=(10, 8))  # Width, Height

    plt.scatter(np.array(occlusion_scores_roi), error_roi, label="regional_occlusion")
    plt.scatter(occlusion_scores_oi, error_oi, label="occlusion_index")
    plt.scatter(occlusion_scores_ci, error_ci, label="crowd_index")

    plt.xlabel("Occlusion Score")
    plt.ylabel("{}".format(" ".join(error_type.split("_")[:-1]).upper()))
    plt.legend(title="k", loc='best')  # You can specify the legend location
    #plt.title("3DPW")
    plt.tight_layout()

    plt.savefig("./ochuman/experiment_figures/experiment1_{}_{}.png".format(method, error_type))


    # Show the plot
    plt.show()
    plt.close()


def experiment4(): # Body Segment OI vs ROI

    index_path = regional_occlusion_anno_path #regional_occlusion_anno_path#occlusion_index_anno_path
    category = "occlusion"
    algorithm = "regional_occlusion" # "occlusion_index" # "regional_occlusion"

    torso_weights_handcrafted = torso_weights_kocabas.copy()
    torso_weights_handcrafted = list(torso_weights_handcrafted.values())
    torso_weights_handcrafted = normalize_to_sum_one(torso_weights_handcrafted)

    upper_body_weights_handcrafted = upper_body_weights_kocabas.copy()
    upper_body_weights_handcrafted = list(upper_body_weights_handcrafted.values())
    upper_body_weights_handcrafted = normalize_to_sum_one(upper_body_weights_handcrafted)

    lower_body_weights_handcrafted = lower_body_weights_kocabas.copy()
    lower_body_weights_handcrafted = list(lower_body_weights_handcrafted.values())
    lower_body_weights_handcrafted = normalize_to_sum_one(lower_body_weights_handcrafted)

    right_body_weights_handcrafted = right_body_weights_kocabas.copy()
    right_body_weights_handcrafted = list(right_body_weights_handcrafted.values())
    right_body_weights_handcrafted = normalize_to_sum_one(right_body_weights_handcrafted)

    left_body_weights_handcrafted = left_body_weights_kocabas.copy()
    left_body_weights_handcrafted = list(left_body_weights_handcrafted.values())
    left_body_weights_handcrafted = normalize_to_sum_one(left_body_weights_handcrafted)


    occlusion_list_torso = process_occlusion_data(index_path, category, torso_weights_handcrafted, algorithm=algorithm)
    occlusion_list_upper_body = process_occlusion_data(index_path, category, upper_body_weights_handcrafted, algorithm=algorithm)
    occlusion_list_lower_body = process_occlusion_data(index_path, category, lower_body_weights_handcrafted, algorithm=algorithm)
    occlusion_list_left_body = process_occlusion_data(index_path, category, left_body_weights_handcrafted, algorithm=algorithm)
    occlusion_list_right_body = process_occlusion_data(index_path, category, right_body_weights_handcrafted, algorithm=algorithm)    
    
    error_torso, occlusion_scores_torso, missing_predictions_torso = calculate_error(error_path=error_path, 
                                                                               occlusion_path=anno_root, 
                                                                               occlusion_list=occlusion_list_torso, 
                                                                               k=len(occlusion_list_torso), 
                                                                               error_algorithm=final_error_algorithm,
                                                                               drop_zero_occ=True)

    error_upper_body, occlusion_scores_upper_body, missing_predictions_upper_body = calculate_error(error_path=error_path, 
                                                                            occlusion_path=anno_root, 
                                                                            occlusion_list=occlusion_list_upper_body, 
                                                                            k=len(occlusion_list_upper_body), 
                                                                            error_algorithm=final_error_algorithm,
                                                                            drop_zero_occ=True)
    
    error_lower_body, occlusion_scores_lower_body, missing_predictions_lower_body = calculate_error(error_path=error_path, 
                                                                            occlusion_path=anno_root, 
                                                                            occlusion_list=occlusion_list_lower_body, 
                                                                            k=len(occlusion_list_lower_body), 
                                                                            error_algorithm=final_error_algorithm,
                                                                            drop_zero_occ=True)
    
    error_left_body, occlusion_scores_left_body, missing_predictions_lower_body = calculate_error(error_path=error_path, 
                                                                            occlusion_path=anno_root, 
                                                                            occlusion_list=occlusion_list_left_body, 
                                                                            k=len(occlusion_list_left_body), 
                                                                            error_algorithm=final_error_algorithm,
                                                                            drop_zero_occ=True)
    
    error_right_body, occlusion_scores_right_body, missing_predictions_lower_body = calculate_error(error_path=error_path, 
                                                                            occlusion_path=anno_root, 
                                                                            occlusion_list=occlusion_list_right_body, 
                                                                            k=len(occlusion_list_right_body), 
                                                                            error_algorithm=final_error_algorithm,
                                                                            drop_zero_occ=True)
    

    


    print(len(error_torso))
    print(len(error_upper_body))
    print(len(error_lower_body))
    print(len(error_left_body))
    print(len(error_right_body))

    #print(occlusion_list_upper_body[:100])

    error_torso = np.array(error_torso)
    error_upper_body = np.array(error_upper_body)
    error_lower_body = np.array(error_lower_body)
    error_left_body = np.array(error_left_body)
    error_right_body = np.array(error_right_body)

    error_torso = error_torso[:, 0]
    error_torso = running_mean(error_torso)
    
    error_upper_body = error_upper_body[:, 0]
    error_upper_body = running_mean(error_upper_body)

    error_lower_body = error_lower_body[:, 0]
    error_lower_body = running_mean(error_lower_body)

    error_left_body = error_left_body[:, 0]
    error_left_body = running_mean(error_left_body)
    
    error_right_body = error_right_body[:, 0]
    error_right_body = running_mean(error_right_body)

    occ_levels = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for occ_level in occ_levels:

        error_indices = closest_elements_indices([occlusion_scores_torso, occlusion_scores_upper_body, occlusion_scores_lower_body, occlusion_scores_left_body, occlusion_scores_right_body], occ_level)

        print(error_indices)
        print("{:.2f}".format(error_torso[error_indices[0]]))    
        print("{:.2f}".format(error_upper_body[error_indices[1]]))
        print("{:.2f}".format(error_lower_body[error_indices[2]]))
        print("{:.2f}".format(error_left_body[error_indices[3]]))
        print("{:.2f}".format(error_right_body[error_indices[4]]))
        print("############################################")
            


    ####################################################

    plt.figure(figsize=(10, 8))  # Width, Height

    plt.scatter(np.array(occlusion_scores_torso) , error_torso, label="Torso: {}".format(len(error_torso)))
    plt.scatter(np.array(occlusion_scores_upper_body) , error_upper_body, label="Upper Body: {}".format(len(error_upper_body)))
    plt.scatter(np.array(occlusion_scores_lower_body) , error_lower_body, label="Lower Body: {}".format(len(error_lower_body)))
    plt.scatter(np.array(occlusion_scores_left_body) , error_left_body, label="Left Body: {}".format(len(error_left_body)))
    plt.scatter(np.array(occlusion_scores_right_body) , error_right_body, label="Right Body: {}".format(len(error_right_body)))

    plt.xlabel("Occlusion Score")
    plt.ylabel("Running Mean {}".format(" ".join(error_type.split("_")[:-2]).upper()))
    plt.legend(title="Groups", loc='best')  # You can specify the legend location
    #plt.title("3DPW: HandCrafted Subsets")
    plt.savefig("./agora/experiment_figures/experiment4_{}_{}_{}.png".format(method, error_type, algorithm))
    plt.tight_layout()

    # Show the plot
    plt.show()
    plt.close()

def experiment5(k = 500): # learn weights from ROMP top k sample

    torso_weights_handcrafted = torso_weights.copy()
    torso_weights_handcrafted = list(torso_weights_handcrafted.values())
    torso_weights_handcrafted = normalize_to_sum_one(torso_weights_handcrafted)

    upper_body_weights_handcrafted = upper_body_weights.copy()
    upper_body_weights_handcrafted = list(upper_body_weights_handcrafted.values())
    upper_body_weights_handcrafted = normalize_to_sum_one(upper_body_weights_handcrafted)

    lower_body_weights_handcrafted = lower_body_weights.copy()
    lower_body_weights_handcrafted = list(lower_body_weights_handcrafted.values())
    lower_body_weights_handcrafted = normalize_to_sum_one(lower_body_weights_handcrafted)

    # occlusion values of the most challanging top k samples from ROMP
    pseudo_weight, joint_names = generate_weights(error_path=error_path, occlusion_path=regional_occlusion_anno_path, k=k, modified=False) 
    pseudo_weight = normalize_to_sum_one(pseudo_weight)

    ####################################################
    current_weight = {}
    #print(joint_names)

    for joint_id, joint_name in enumerate(joint_names):
        current_weight[joint_name] = pseudo_weight[joint_id]

    new_weights = {}

    extreme_merge = False 

    for joint_id, joint_name in enumerate(joint_names):
        if extreme_merge:
            if joint_name in right_upper_joint_names:
                new_weights["rightUpper"] = np.sum([current_weight[joint_nm] for joint_nm in right_upper_joint_names]) / len(right_upper_joint_names)

            elif joint_name in left_upper_joint_names:
                new_weights["leftUpper"] = np.sum([current_weight[joint_nm] for joint_nm in left_upper_joint_names]) / len(left_upper_joint_names)

            elif joint_name in right_lower_joint_names:
                new_weights["rightLower"] = np.sum([current_weight[joint_nm] for joint_nm in right_lower_joint_names]) / len(right_lower_joint_names)

            elif joint_name in left_lower_joint_names:
                new_weights["leftLower"] = np.sum([current_weight[joint_nm] for joint_nm in left_lower_joint_names]) / len(left_lower_joint_names)

            elif joint_name in torso_joint_names:
                new_weights["torso"] = np.sum([current_weight[joint_nm] for joint_nm in torso_joint_names]) / len(torso_joint_names)

            elif joint_name in head_joint_names:
                new_weights["head"] = np.sum([current_weight[joint_nm] for joint_nm in head_joint_names]) / len(head_joint_names)
            else:
                new_weights[joint_name] = current_weight[joint_name]
        else:
            if joint_name in ["rightHand", "rightHandIndex1"]:
                new_weights["rightHand"] = (current_weight["rightHand"] + current_weight["rightHandIndex1"]) /2

            elif joint_name in ["rightFoot", "rightToeBase"]:
                new_weights["rightFoot"] = (current_weight["rightFoot"] + current_weight["rightToeBase"]) /2

            elif joint_name in ["leftHand", "leftHandIndex1"]:
                new_weights["leftHand"] = (current_weight["leftHand"] + current_weight["leftHandIndex1"]) /2

            elif joint_name in ["leftFoot", "leftToeBase"]:
                new_weights["leftFoot"] = (current_weight["leftFoot"] + current_weight["leftToeBase"]) /2

            elif joint_name in ["head", "neck"]:
                new_weights["head"] = (current_weight["head"] + current_weight["neck"]) /2

            elif joint_name in ["spine", "spine1", "spine2"]:
                new_weights["torso"] = (current_weight["spine"] + current_weight["spine1"] + current_weight["spine2"]) / 3
            else:
                new_weights[joint_name] = current_weight[joint_name]

    #new_weights = current_weight # this removes all merges

    plt.figure(figsize=(10, 8))  # Width, Height

    plt.bar(new_weights.keys(), new_weights.values())

    plt.xlabel("Joint Name")
    plt.ylabel("Weight (%)", )
    plt.xticks(rotation=90)
    plt.title("3DPW: Learned Weights from {} (k:{})".format(method, k))
    plt.tight_layout()

    # Show the plot
    plt.show()
    plt.close()

    ####################################################

    occlusion_list_torso = process_occlusion_data(regional_occlusion_anno_path, category, torso_weights_handcrafted, algorithm="regional_occlusion")
    occlusion_list_upper_body = process_occlusion_data(regional_occlusion_anno_path, category, upper_body_weights_handcrafted, algorithm="regional_occlusion")
    occlusion_list_lower_body = process_occlusion_data(regional_occlusion_anno_path, category, lower_body_weights_handcrafted, algorithm="regional_occlusion")
    occlusion_list_learned = process_occlusion_data(regional_occlusion_anno_path, category, pseudo_weight, algorithm="regional_occlusion")

    error_torso, occlusion_scores_torso, missing_predictions_torso = calculate_error(error_path=error_path, 
                                                                               occlusion_path=anno_root, 
                                                                               occlusion_list=occlusion_list_torso, 
                                                                               k=len(occlusion_list_torso), 
                                                                               error_algorithm=final_error_algorithm,
                                                                               drop_zero_occ=True)
    
    error_upper_body, occlusion_scores_upper_body, missing_predictions_upper_body = calculate_error(error_path=error_path, 
                                                                            occlusion_path=anno_root, 
                                                                            occlusion_list=occlusion_list_upper_body, 
                                                                            k=len(occlusion_list_upper_body), 
                                                                            error_algorithm=final_error_algorithm,
                                                                            drop_zero_occ=True)
    
    error_lower_body, occlusion_scores_lower_body, missing_predictions_lower_body = calculate_error(error_path=error_path, 
                                                                            occlusion_path=anno_root, 
                                                                            occlusion_list=occlusion_list_lower_body, 
                                                                            k=len(occlusion_list_lower_body), 
                                                                            error_algorithm=final_error_algorithm,
                                                                            drop_zero_occ=True)
    
    error_learned, occlusion_scores_learned, missing_predictions_learned = calculate_error(error_path=error_path, 
                                                                            occlusion_path=anno_root, 
                                                                            occlusion_list=occlusion_list_learned, 
                                                                            k=len(occlusion_list_learned), 
                                                                            error_algorithm=final_error_algorithm,
                                                                            drop_zero_occ=True)

    error_torso = np.array(error_torso)
    error_upper_body = np.array(error_upper_body)
    error_lower_body = np.array(error_lower_body)
    error_learned = np.array(error_learned)

    error_torso = error_torso[:, 0]
    error_torso = running_mean(error_torso)

    error_upper_body = error_upper_body[:, 0]
    error_upper_body = running_mean(error_upper_body)

    error_lower_body = error_lower_body[:, 0]
    error_lower_body = running_mean(error_lower_body)

    error_learned = error_learned[:, 0]
    error_learned = running_mean(error_learned)

    ####################################################

    x_torso = np.arange(len(error_torso))

    x_torso = [i/len(error_torso) for i in x_torso]

    x_upper_body = np.arange(len(error_upper_body))

    x_upper_body = [i/len(error_upper_body) for i in x_upper_body]

    x_lower_body = np.arange(len(error_lower_body))

    x_lower_body = [i/len(error_lower_body) for i in x_lower_body]

    x_learned = np.arange(len(error_learned))

    x_learned = [i/len(error_learned) for i in x_learned]

    ####################################################

    plt.figure(figsize=(10, 8))  # Width, Height

    plt.scatter(np.array(occlusion_scores_torso) / 100, error_torso, label="torso: {}".format(len(error_torso)))
    plt.scatter(np.array(occlusion_scores_upper_body) / 100, error_upper_body, label="upper body: {}".format(len(error_upper_body)))
    plt.scatter(np.array(occlusion_scores_lower_body) / 100, error_lower_body, label="lower body: {}".format(len(error_lower_body)))
    plt.scatter(np.array(occlusion_scores_learned) / 100, error_learned, label="learned from {} k:{}".format(method, k))

    plt.xlabel("Occlusion Score")
    plt.ylabel("MPJPE")
    plt.legend(title="k", loc='best')  # You can specify the legend location
    plt.title("3DPW: Learned from {} Subset".format(method))
    plt.tight_layout()

    # Show the plot
    plt.show()
    plt.close()

def experiment6(): # per joint error per joint visibility
    _, joint_names = generate_weights(error_path=error_path, occlusion_path=regional_occlusion_anno_path, k=10, modified=False) 
    joint_names = list(joint_names)
    n_joints = len(joint_names)
    
    torso_merge_indices = [joint_names.index("spine"), joint_names.index("spine1"), joint_names.index("spine2")]
    head_merge_indices = [joint_names.index("head"), joint_names.index("neck")]
    right_hand_indices = [joint_names.index("rightHand"), joint_names.index("rightHandIndex1")]
    right_foot_indices = [joint_names.index("rightFoot"), joint_names.index("rightToeBase")]
    left_hand_indices = [joint_names.index("leftHand"), joint_names.index("leftHandIndex1")]
    left_foot_indices = [joint_names.index("leftFoot"), joint_names.index("leftToeBase")]
    
    new_joints = ['overall', 'rightHand', 'rightUpLeg', 'leftArm', 
                  'leftLeg', 'leftFoot', 'torso', 'leftShoulder', 
                  'rightShoulder', 'rightFoot', 'head', 'rightArm', 
                  'leftHand', 'rightLeg', 'leftForeArm', 'rightForeArm', 
                  'leftUpLeg', 'hips']
    """
    n_joints = len(new_joints)
    joint_names = new_joints
    """
    errors = np.zeros((n_joints, n_joints))

    all_data_occlusion_sorted = process_occlusion_data(regional_occlusion_anno_path, category, list(uniform_weights.values()), algorithm="regional_occlusion")

    all_data_error, _, _ = calculate_error(error_path=error_path, 
                                            occlusion_path=anno_root, 
                                            occlusion_list=all_data_occlusion_sorted, 
                                            k=len(all_data_occlusion_sorted), 
                                            error_algorithm=final_error_algorithm,
                                            drop_zero_occ=True)

    all_data_overall_error = np.mean(all_data_error, axis=0)
    #print(all_data_overall_error)

    for joint_id, joint_name in enumerate(joint_names):
        current_occlusion_weights = uniform_weights.copy()
        current_occlusion_weights["overall"] = 0.0
        current_occlusion_weights[joint_name] = 1.0

        current_occlusion_weights = list(current_occlusion_weights.values())
        current_occlusion_weights = normalize_to_sum_one(current_occlusion_weights)

        occlusion_list_current = process_occlusion_data(regional_occlusion_anno_path, category, current_occlusion_weights, algorithm="regional_occlusion")
        
        #occlusion_list_current = occlusion_list_current[:50]

        error_current, occlusion_scores_current, missing_predictions_current = calculate_error(error_path=error_path, 
                                                                            occlusion_path=anno_root, 
                                                                            occlusion_list=occlusion_list_current, 
                                                                            k=len(occlusion_list_current), 
                                                                            error_algorithm=final_error_algorithm,
                                                                            drop_zero_occ=True)
        
        print(joint_name, len(error_current))


        error_current = np.mean(error_current, axis=0)

        #error_current -= all_data_overall_error

        torso_mean = np.mean(error_current[torso_merge_indices])
        head_mean = np.mean(error_current[head_merge_indices])
        rightHand_mean = np.mean(error_current[right_hand_indices])
        leftHand_mean = np.mean(error_current[left_hand_indices])
        rightFoot_mean = np.mean(error_current[right_foot_indices])
        leftFoot_mean = np.mean(error_current[left_foot_indices])

        new_joints_error_current = np.zeros_like(new_joints, dtype=np.float32)

        for sub_joint_id, sub_joint_name in enumerate(joint_names):
            if sub_joint_name in ["rightHand", "rightHandIndex1"]:
                new_joints_error_current[new_joints.index("rightHand")] = rightHand_mean

            elif sub_joint_name in ["rightFoot", "rightToeBase"]:
                new_joints_error_current[new_joints.index("rightFoot")] = rightFoot_mean

            elif sub_joint_name in ["leftHand", "leftHandIndex1"]:
                new_joints_error_current[new_joints.index("leftHand")] = leftHand_mean

            elif sub_joint_name in ["leftFoot", "leftToeBase"]:
                new_joints_error_current[new_joints.index("leftFoot")] = leftFoot_mean

            elif sub_joint_name in ["head", "neck"]:
                new_joints_error_current[new_joints.index("head")] = head_mean

            elif sub_joint_name in ["spine", "spine1", "spine2"]:
                new_joints_error_current[new_joints.index("torso")] = torso_mean
            else:
                new_joints_error_current[new_joints.index(sub_joint_name)] = error_current[sub_joint_id]

        """
        overall_error = new_joints_error_current[0]

        #new_joints_error_current -= all_data_overall_occlusion
        
        error_by_joint = zip(new_joints[1:], new_joints_error_current[1:])

        error_by_joint = sorted(error_by_joint, key=lambda x: x[1], reverse=True)

        joints_sorted = list(map(lambda x: x[0], error_by_joint))
        errors_sorted = list(map(lambda x: x[1], error_by_joint))
        
        
        fig, ax = plt.subplots()

        ax.barh(joints_sorted, errors_sorted, color ='maroon')
        ax.set_xlabel("MPJPE")
        ax.set_ylabel("Joints")
        ax.invert_yaxis()
        ax.axvline(x = overall_error, color = 'b', label = 'mean')
        ax.set_title(joint_name)
        plt.legend()
        plt.tight_layout()
        plt.show()
        """



        #if joint_name == "leftLeg":

        #    sys.exit()


        errors[joint_id] = error_current
        """
        torso_mean = np.mean(error_current[torso_merge_indices])
        head_mean = np.mean(error_current[head_merge_indices])
        rightHand_mean = np.mean(error_current[right_hand_indices])
        leftHand_mean = np.mean(error_current[left_hand_indices])
        rightFoot_mean = np.mean(error_current[right_foot_indices])
        leftFoot_mean = np.mean(error_current[left_foot_indices])

        for new_joint_name in 

        error_current = replace_mean(array=error_current, merge_indices=torso_merge_indices, new_index=new_joints.index("torso"))
        error_current = replace_mean(array=error_current, merge_indices=head_merge_indices, new_index=new_joints.index("head"))
        error_current = replace_mean(array=error_current, merge_indices=right_hand_indices, new_index=new_joints.index("rightHand"))
        error_current = replace_mean(array=error_current, merge_indices=left_hand_indices, new_index=new_joints.index("leftHand"))
        error_current = replace_mean(array=error_current, merge_indices=right_foot_indices, new_index=new_joints.index("rightFoot"))
        error_current = replace_mean(array=error_current, merge_indices=left_foot_indices, new_index=new_joints.index("leftFoot"))


        if joint_name in ["rightHand", "rightHandIndex1"]:
            errors[new_joints.index("rightHand")] = error_current / 2

        elif joint_name in ["rightFoot", "rightToeBase"]:
            errors[new_joints.index("rightFoot")] = error_current / 2

        elif joint_name in ["leftHand", "leftHandIndex1"]:
            errors[new_joints.index("leftHand")] = error_current / 2

        elif joint_name in ["leftFoot", "leftToeBase"]:
            errors[new_joints.index("leftFoot")] = error_current / 2

        elif joint_name in ["head", "neck"]:
            errors[new_joints.index("head")] = error_current / 2

        elif joint_name in ["spine", "spine1", "spine2"]:
            errors[new_joints.index("torso")] = error_current / 3 
        else:
            errors[new_joints.index(joint_name)] = error_current

        """


    # Plotting the heatmap
    plt.imshow(errors, cmap='inferno', interpolation='nearest')

    # Adding a colorbar
    plt.colorbar()
    plt.ylabel("Occluded Joint")
    plt.yticks(np.arange(len(joint_names)), joint_names)    
    plt.xticks(np.arange(len(joint_names)), joint_names, rotation=90)
    
    plt.xlabel("MPJPE per Joint")
    plt.tight_layout()

    # Display the plot
    plt.show()

def experiment6_0():

    joint_names_kocabas = list(uniform_weights_kocabas.keys())[1:]
    joint_names_smpl = list(uniform_weights.keys())[1:]

    n_joints = len(joint_names_kocabas)

    errors = np.zeros((n_joints, n_joints))

    for joint_id, joint_name in enumerate(joint_names_kocabas):
        current_occlusion_weights = uniform_weights_kocabas.copy()
        current_occlusion_weights["overall"] = 0.0
        current_occlusion_weights[joint_name] = 1.0

        current_occlusion_weights = list(current_occlusion_weights.values())
        current_occlusion_weights = normalize_to_sum_one(current_occlusion_weights)

        occlusion_list_current = process_occlusion_data(regional_occlusion_anno_path, category, current_occlusion_weights, algorithm="regional_occlusion")
        
        occlusion_list_current = occlusion_list_current[:50]

        error_current, occlusion_scores_current, missing_predictions_current = calculate_error(error_path=error_path, 
                                                                            occlusion_path=anno_root, 
                                                                            occlusion_list=occlusion_list_current, 
                                                                            k=len(occlusion_list_current), 
                                                                            error_algorithm=final_error_algorithm,
                                                                            drop_zero_occ=True)
        

        error_current = np.mean(error_current, axis=0)[1:]

        """
        fig, ax = plt.subplots()

        ax.barh(joint_names_smpl, error_current, color ='maroon')
        ax.set_xlabel("MPJPE")
        ax.set_ylabel("Joints")
        ax.invert_yaxis()
        ax.axvline(x = error_current[0], color = 'b', label = 'mean')
        ax.set_title(joint_name)
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
        


        errors[joint_id] = error_current

    import re
    for i in range(n_joints):
        words = joint_names_smpl[i].split('_')
        formatted_string = ' '.join(word.capitalize() for word in words)
        joint_names_smpl[i] = formatted_string

        formatted_string = re.sub(r'([a-z])([A-Z])', r'\1 \2', joint_names_kocabas[i])
        # Capitalize the first letter of each word
        formatted_string = ' '.join(word.capitalize() for word in formatted_string.split())

        joint_names_kocabas[i] = formatted_string
    
    
    
    # Plotting the heatmap
    plt.imshow(errors, cmap='inferno', interpolation='nearest')

    # Adding a colorbar
    plt.colorbar()
    plt.ylabel("Occluded Body Segment")
    plt.yticks(np.arange(len(joint_names_kocabas)), joint_names_kocabas)    
    plt.xticks(np.arange(len(joint_names_smpl)), joint_names_smpl, rotation=90)
    
    plt.xlabel("{} per Joint".format(" ".join(error_type.split("_")[:-2]).upper()))
    plt.tight_layout()

    plt.savefig("./agora/experiment_figures/experiment6_{}_{}_{}.png".format(method, error_type, "regional_occlusion"))


    # Display the plot
    plt.show()

def experiment7(): # TODO self occlusion
    pass

def experiment8(): # TODO truncation
    weights = uniform_weights
    weights = list(weights.values())
    weights = normalize_to_sum_one(weights)

    occlusion_list_roi = process_truncation_data(regional_occlusion_anno_path, weights)
    
    k = len(occlusion_list_roi)
    error_roi, occlusion_scores_roi, missing_predictions_roi = calculate_error(error_path=error_path, occlusion_path=anno_root, occlusion_list=occlusion_list_roi, k=k, error_algorithm=final_error_algorithm)

    error_roi = np.array(error_roi)

    error_roi = error_roi[:, 0]

    plt.figure(figsize=(10, 8))  # Width, Height

    plt.scatter(np.array(occlusion_scores_roi)/100, error_roi, label="regional_occlusion")

    plt.xlabel("Occlusion Score")
    plt.ylabel("MPJPE")
    plt.legend(title="k", loc='best')  # You can specify the legend location
    plt.title("3DPW: All Dataset")
    plt.tight_layout()

    # Show the plot
    plt.show()
    plt.close()

def experiment9(k=500):

    # occlusion values of the most challanging top k samples from ROMP
    pseudo_weight, joint_names = generate_weights(error_path=error_path, occlusion_path=regional_occlusion_anno_path, k=k, modified=False) 
    pseudo_weight = normalize_to_sum_one(pseudo_weight)

    # occlusion values of the most challanging top k samples from ROMP
    pseudo_weight_modified, joint_names = generate_weights(error_path=error_path, occlusion_path=regional_occlusion_anno_path, k=k, modified=True) 
    pseudo_weight_modified = normalize_to_sum_one(pseudo_weight_modified)

    ####################################################
    current_weight = {}
    current_weight_modified = {}
    #print(joint_names)

    for joint_id, joint_name in enumerate(joint_names):
        current_weight[joint_name] = pseudo_weight[joint_id]
        current_weight_modified[joint_name] = pseudo_weight_modified[joint_id]
        
    new_weights = {}
    new_weights_modified = {} 

    for joint_id, joint_name in enumerate(joint_names):
        if joint_name in ["rightHand", "rightHandIndex1"]:
            new_weights["rightHand"] = (current_weight["rightHand"] + current_weight["rightHandIndex1"]) /2
            new_weights_modified["rightHand"] = (current_weight_modified["rightHand"] + current_weight_modified["rightHandIndex1"]) /2

        elif joint_name in ["rightFoot", "rightToeBase"]:
            new_weights["rightFoot"] = (current_weight["rightFoot"] + current_weight["rightToeBase"]) /2
            new_weights_modified["rightFoot"] = (current_weight_modified["rightFoot"] + current_weight_modified["rightToeBase"]) /2

        elif joint_name in ["leftHand", "leftHandIndex1"]:
            new_weights["leftHand"] = (current_weight["leftHand"] + current_weight["leftHandIndex1"]) /2
            new_weights_modified["leftHand"] = (current_weight_modified["leftHand"] + current_weight_modified["leftHandIndex1"]) /2

        elif joint_name in ["leftFoot", "leftToeBase"]:
            new_weights["leftFoot"] = (current_weight["leftFoot"] + current_weight["leftToeBase"]) /2
            new_weights_modified["leftFoot"] = (current_weight_modified["leftFoot"] + current_weight_modified["leftToeBase"]) /2

        elif joint_name in ["head", "neck"]:
            new_weights["head"] = (current_weight["head"] + current_weight["neck"]) /2
            new_weights_modified["head"] = (current_weight_modified["head"] + current_weight_modified["neck"]) /2

        elif joint_name in ["spine", "spine1", "spine2"]:
            new_weights["torso"] = (current_weight["spine"] + current_weight["spine1"] + current_weight["spine2"]) / 3
            new_weights_modified["torso"] = (current_weight_modified["spine"] + current_weight_modified["spine1"] + current_weight_modified["spine2"]) / 3

        else:
            new_weights[joint_name] = current_weight[joint_name]
            new_weights_modified[joint_name] = current_weight_modified[joint_name]

    #new_weights = current_weight # this removes all merges

    new_keys = ["head", "torso", "hips", "rightHand", "leftHand", "rightForeArm", "leftForeArm",
                "rightArm", "leftArm", "rightShoulder", "leftShoulder", "rightUpLeg", "leftUpLeg", 
                "rightLeg", "leftLeg", "rightFoot", "leftFoot"]

    plt.figure(figsize=(10, 8))  # Width, Height

    # Set the width of the bars
    bar_width = 0.35

    # Set the position of the bars on the x-axis
    index = np.arange(len(new_weights_modified.keys())-1)
    
    new_weights_values = [new_weights[joint_name] for joint_name in new_keys]
    new_weights_modified_values = [new_weights_modified[joint_name] for joint_name in new_keys]

    plt.bar(index, new_weights_values, bar_width, label='Normal Weights',  align='center')
    plt.bar(index+bar_width, new_weights_modified_values, bar_width, label='Modified Weights', align='center')
    plt.axhline(y = new_weights["overall"], color = 'r', linestyle = '-', label="Normal Weights Mean") 
    plt.axhline(y = new_weights_modified["overall"], color = 'b', linestyle = '-', label="Modified Weights Mean") 

    plt.xlabel("Joint Name")
    plt.ylabel("Weight (%)", )
    plt.xticks(index + bar_width / 2, new_keys, rotation=90)  # Center x ticks between the two bar groups
    plt.title("3DPW: Learned Weights from {} (k:{})".format(method, k))
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()
    plt.close()

    ####################################################

    occlusion_list_learned = process_occlusion_data(regional_occlusion_anno_path, category, pseudo_weight, algorithm="regional_occlusion")
    occlusion_list_learned_modified = process_occlusion_data(regional_occlusion_anno_path, category, pseudo_weight_modified, algorithm="regional_occlusion")
    
    final_error_algorithm = "standard" # standard, keypoint_visibility, truncation, region_visibility

    # Normal weights x Normal error
    error_learned_nn, occlusion_scores_learned_nn, missing_predictions_learned = calculate_error(error_path=error_path, 
                                                                            occlusion_path=anno_root, 
                                                                            occlusion_list=occlusion_list_learned, 
                                                                            k=len(occlusion_list_learned), 
                                                                            error_algorithm=final_error_algorithm,
                                                                            drop_zero_occ=True)
    
    # Normal weights x Modified Error
    final_error_algorithm = "region_visibility" # standard, keypoint_visibility, truncation, region_visibility

    error_learned_nm, occlusion_scores_learned_nm, missing_predictions_learned = calculate_error(error_path=error_path, 
                                                                        occlusion_path=anno_root, 
                                                                        occlusion_list=occlusion_list_learned, 
                                                                        k=len(occlusion_list_learned), 
                                                                        error_algorithm=final_error_algorithm,
                                                                        drop_zero_occ=True)

    # Modified Weights x Normal Error
    final_error_algorithm = "standard" # standard, keypoint_visibility, truncation, region_visibility
    error_learned_mn, occlusion_scores_learned_mn, missing_predictions_learned = calculate_error(error_path=error_path, 
                                                                        occlusion_path=anno_root, 
                                                                        occlusion_list=occlusion_list_learned_modified, 
                                                                        k=len(occlusion_list_learned_modified), 
                                                                        error_algorithm=final_error_algorithm,
                                                                        drop_zero_occ=True)
    
    # Modified Weights x Modified Error
    final_error_algorithm = "region_visibility" # standard, keypoint_visibility, truncation, region_visibility

    error_learned_mm, occlusion_scores_learned_mm, missing_predictions_learned = calculate_error(error_path=error_path, 
                                                                        occlusion_path=anno_root, 
                                                                        occlusion_list=occlusion_list_learned_modified, 
                                                                        k=len(occlusion_list_learned_modified), 
                                                                        error_algorithm=final_error_algorithm,
                                                                        drop_zero_occ=True)

    error_learned_nn = np.array(error_learned_nn)
    error_learned_nn = error_learned_nn[:, 0]
    error_learned_nn = running_mean(error_learned_nn)

    error_learned_nm = np.array(error_learned_nm)
    error_learned_nm = error_learned_nm[:, 0]
    error_learned_nm = running_mean(error_learned_nm)

    error_learned_mn = np.array(error_learned_mn)
    error_learned_mn = error_learned_mn[:, 0]
    error_learned_mn = running_mean(error_learned_mn)

    error_learned_mm = np.array(error_learned_mm)
    error_learned_mm = error_learned_mm[:, 0]
    error_learned_mm = running_mean(error_learned_mm)


    ####################################################

    x_learned_nn = np.arange(len(error_learned_nn))

    x_learned_nn = [i/len(error_learned_nn) for i in x_learned_nn]
    ##

    x_learned_nm = np.arange(len(error_learned_nm))

    x_learned_nm = [i/len(error_learned_nm) for i in x_learned_nm]
    ##

    x_learned_mn = np.arange(len(error_learned_mn))

    x_learned_mn = [i/len(error_learned_mn) for i in x_learned_mn]
    ##

    x_learned_mm = np.arange(len(error_learned_mm))

    x_learned_mm = [i/len(error_learned_mm) for i in x_learned_mm]

    ####################################################

    plt.figure(figsize=(10, 8))  # Width, Height

    plt.scatter(np.array(occlusion_scores_learned_nn) / 100, error_learned_nn, label="Weights:Normal, Error:Normal")
    plt.scatter(np.array(occlusion_scores_learned_nm) / 100, error_learned_nm, label="Weights:Normal, Error:Modified")
    plt.scatter(np.array(occlusion_scores_learned_mn) / 100, error_learned_mn, label="Weights:Modified, Error:Normal")
    plt.scatter(np.array(occlusion_scores_learned_mm) / 100, error_learned_mm, label="Weights:Modified, Error:Modified")

    plt.xlabel("Occlusion Score")
    plt.ylabel("MPJPE")
    plt.legend(title="k", loc='best')  # You can specify the legend location
    plt.title("3DPW: Learned from {} Subset (k:{})".format(method, k))
    plt.tight_layout()

    # Show the plot
    plt.show()
    plt.close()

def experiment9_1():

    weights = uniform_weights_kocabas
    weights = list(weights.values())
    weights = normalize_to_sum_one(weights)

    error_here = "pa_mpjpe_14_dec"

    occlusion_list_roi = process_occlusion_data(regional_occlusion_anno_path, category, weights, algorithm="regional_occlusion")

    error_romp_standard, occlusion_scores_romp_standard, missing_error_roipredictions_roi = calculate_error(error_path="./methods/romp_results/agora/{}.json".format(error_here), occlusion_path=anno_root, occlusion_list=occlusion_list_roi, k=len(occlusion_list_roi), error_algorithm=final_error_algorithm)
    error_bev_standard, occlusion_scores_bev_standard, missing_error_roipredictions_roi = calculate_error(error_path="./methods/bev_results/agora/{}.json".format(error_here), occlusion_path=anno_root, occlusion_list=occlusion_list_roi, k=len(occlusion_list_roi), error_algorithm=final_error_algorithm)

    error_romp_modified, occlusion_scores_romp_modified, missing_error_roipredictions_roi = calculate_error(error_path="./methods/romp_results/agora/{}.json".format(error_here), occlusion_path=anno_root, occlusion_list=occlusion_list_roi, k=len(occlusion_list_roi), error_algorithm="region_visibility")
    error_bev_modified, occlusion_scores_bev_modified, missing_error_roipredictions_roi = calculate_error(error_path="./methods/bev_results/agora/{}.json".format(error_here), occlusion_path=anno_root, occlusion_list=occlusion_list_roi, k=len(occlusion_list_roi), error_algorithm="region_visibility")

    error_romp_standard = np.array(error_romp_standard)
    error_bev_standard = np.array(error_bev_standard)
    error_romp_modified = np.array(error_romp_modified)
    error_bev_modified = np.array(error_bev_modified)

    error_romp_standard = error_romp_standard[:, 0]
    error_romp_standard = running_mean(error_romp_standard)

    error_bev_standard = error_bev_standard[:, 0]
    error_bev_standard = running_mean(error_bev_standard)

    error_romp_modified = error_romp_modified[:, 0]
    error_romp_modified = running_mean(error_romp_modified)

    error_bev_modified = error_bev_modified[:, 0]
    error_bev_modified = running_mean(error_bev_modified)

    occ_levels = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for occ_level in occ_levels:

        error_indices = closest_elements_indices([occlusion_scores_romp_standard, 
                                                  occlusion_scores_bev_standard, 
                                                  occlusion_scores_romp_modified,
                                                  occlusion_scores_bev_modified], 
                                                  occ_level)

        print(error_indices)
        print("romp normal: {:.2f}".format(error_romp_standard[error_indices[0]]))    
        print("bev normal: {:.2f}".format(error_bev_standard[error_indices[1]]))
        print("romp modified: {:.2f}".format(error_romp_modified[error_indices[2]]))
        print("bev modified: {:.2f}".format(error_bev_modified[error_indices[3]]))
        print("############################################")

    plt.figure(figsize=(10, 8))  # Width, Height

    plt.scatter(occlusion_scores_romp_modified, error_romp_modified, label="ROMP: Modified Metric")
    plt.scatter(occlusion_scores_bev_modified, error_bev_modified, label="BEV: Modified Metric")
    plt.scatter(occlusion_scores_romp_standard, error_romp_standard, label="ROMP: Standard Metric")
    plt.scatter(occlusion_scores_bev_standard, error_bev_standard, label="BEV: Standard Metric")
    

    plt.xlabel("Occlusion Score")
    plt.ylabel("Running Mean of {}".format(" ".join(error_type.split("_")[:-2]).upper()))
    plt.legend(loc='best',prop = { "size": 14 })  # You can specify the legend location
    #plt.title("ROMP's Performance on Top 10% Challenging Samples of 3DPW According to Different Indices")
    plt.tight_layout()

    plt.savefig("./agora/experiment_figures/experiment91_{}.png".format(error_here))

    # Show the plot
    plt.show()
    plt.close()


def experiment10(k=150):

    weights_method = "bev"
    weights_error_type = "mpjpe_14_dec" #error_type #"pa_mpjpe_joints"

    weights = uniform_weights_kocabas
    weights = list(weights.values())
    weights = normalize_to_sum_one(weights)

    torso_weights_handcrafted = torso_weights_kocabas.copy()
    torso_weights_handcrafted = list(torso_weights_handcrafted.values())
    torso_weights_handcrafted = normalize_to_sum_one(torso_weights_handcrafted)

    lower_body_weights_handcrafted = lower_body_weights_kocabas.copy()
    lower_body_weights_handcrafted = list(lower_body_weights_handcrafted.values())
    lower_body_weights_handcrafted = normalize_to_sum_one(lower_body_weights_handcrafted)

    upper_body_weights_handcrafted = upper_body_weights_kocabas.copy()
    upper_body_weights_handcrafted = list(upper_body_weights_handcrafted.values())
    upper_body_weights_handcrafted = normalize_to_sum_one(upper_body_weights_handcrafted)

    # occlusion values of the most challanging top k samples from ROMP
    pseudo_weight, joint_names = generate_weights(error_path="./methods/{}_results/{}/{}.json".format(weights_method, dataset, weights_error_type), occlusion_path=regional_occlusion_anno_path, k=k, modified=False) 
    pseudo_weight = normalize_to_sum_one(pseudo_weight)


    ####################################################

    occlusion_list_uniform = process_occlusion_data(regional_occlusion_anno_path, category, weights, algorithm="regional_occlusion")
    occlusion_list_learned = process_occlusion_data(regional_occlusion_anno_path, category, pseudo_weight, algorithm="regional_occlusion")
    occlusion_list_torso = process_occlusion_data(regional_occlusion_anno_path, category, torso_weights_handcrafted, algorithm="regional_occlusion")
    occlusion_list_upper_body = process_occlusion_data(regional_occlusion_anno_path, category, upper_body_weights_handcrafted, algorithm="regional_occlusion")
    occlusion_list_lower_body = process_occlusion_data(regional_occlusion_anno_path, category, lower_body_weights_handcrafted, algorithm="regional_occlusion")

    error_uniform, occlusion_scores_uniform, missing_predictions_uniform = calculate_error(error_path=error_path, 
                                                                               occlusion_path=anno_root, 
                                                                               occlusion_list=occlusion_list_uniform, 
                                                                               k=len(occlusion_list_uniform), 
                                                                               error_algorithm=final_error_algorithm,
                                                                               drop_zero_occ=True)
    
    error_learned, occlusion_scores_learned, missing_predictions_learned = calculate_error(error_path=error_path, 
                                                                            occlusion_path=anno_root, 
                                                                            occlusion_list=occlusion_list_learned, 
                                                                            k=len(occlusion_list_learned), 
                                                                            error_algorithm=final_error_algorithm,
                                                                            drop_zero_occ=True)
    
    error_torso, occlusion_scores_torso, missing_predictions_torso = calculate_error(error_path=error_path, 
                                                                               occlusion_path=anno_root, 
                                                                               occlusion_list=occlusion_list_torso, 
                                                                               k=len(occlusion_list_torso), 
                                                                               error_algorithm=final_error_algorithm,
                                                                               drop_zero_occ=True)

    error_upper_body, occlusion_scores_upper_body, missing_predictions_upper_body = calculate_error(error_path=error_path, 
                                                                            occlusion_path=anno_root, 
                                                                            occlusion_list=occlusion_list_upper_body, 
                                                                            k=len(occlusion_list_upper_body), 
                                                                            error_algorithm=final_error_algorithm,
                                                                            drop_zero_occ=True)
    
    error_lower_body, occlusion_scores_lower_body, missing_predictions_lower_body = calculate_error(error_path=error_path, 
                                                                            occlusion_path=anno_root, 
                                                                            occlusion_list=occlusion_list_lower_body, 
                                                                            k=len(occlusion_list_lower_body), 
                                                                            error_algorithm=final_error_algorithm,
                                                                            drop_zero_occ=True)

    error_uniform = np.array(error_uniform)
    error_learned = np.array(error_learned)
    error_torso = np.array(error_torso)
    error_upper_body = np.array(error_upper_body)
    error_lower_body = np.array(error_lower_body)

    error_uniform = error_uniform[:, 0]
    error_uniform = running_mean(error_uniform)

    error_learned = error_learned[:, 0]
    error_learned = running_mean(error_learned)

    error_torso = error_torso[:, 0]
    error_torso = running_mean(error_torso)
    
    error_upper_body = error_upper_body[:, 0]
    error_upper_body = running_mean(error_upper_body)

    error_lower_body = error_lower_body[:, 0]
    error_lower_body = running_mean(error_lower_body)

    ####################################################

    plt.figure(figsize=(10, 8))  # Width, Height

    
    plt.scatter(np.array(occlusion_scores_torso) , error_torso, label="Torso")
    plt.scatter(np.array(occlusion_scores_upper_body) , error_upper_body, label="Upper Body")
    plt.scatter(np.array(occlusion_scores_lower_body) , error_lower_body, label="Lower Body")
    plt.scatter(np.array(occlusion_scores_uniform), error_uniform, label="uniform: {}".format(len(error_uniform)))
    plt.scatter(np.array(occlusion_scores_learned), error_learned, label="learned from {} k:{}".format(weights_method, k))

    plt.xlabel("Occlusion Score")
    plt.ylabel("MPJPE")
    plt.legend(title="k", loc='best')  # You can specify the legend location
    plt.title("AGORA: Learned from {} Subset".format(method))
    plt.tight_layout()

    # Show the plot
    plt.show()
    plt.close()

#experiment1()
#experiment2()
#experiment3()
#experiment4() 
#experiment5(k=30)
#experiment6()
#experiment6_0()
#experiment7()
#experiment8()
#experiment9(k=1000)
experiment9_1()

#for k_value in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400, 500, 1000]:
#    experiment10(k=k_value)
smpl_joint_names = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', "left_hand", "right_hand"
]
