import numpy as np
import os, json, glob, sys
from datetime import datetime
from operator import itemgetter
import matplotlib.pyplot as plt
from subset_weights import uniform_weights, lower_body_weights, torso_weights, upper_body_weights
import cv2

truncated_seqs = ["courtyard_basketball_00", "courtyard_warmWelcome_00"]#, "downtows_bar_00"]

erroneous_annotation_list = ['courtyard_dancing_01/image_00114.jpg', 'courtyard_dancing_01/image_00106.jpg', 
                             'courtyard_dancing_01/image_00107.jpg', 'courtyard_dancing_01/image_00118.jpg', 
                             'courtyard_dancing_01/image_00108.jpg', 'courtyard_dancing_01/image_00116.jpg', 
                             'courtyard_dancing_01/image_00113.jpg', 'courtyard_dancing_01/image_00119.jpg', 
                             'courtyard_dancing_01/image_00117.jpg', 'courtyard_dancing_01/image_00120.jpg', 
                             'courtyard_dancing_01/image_00111.jpg', 'courtyard_dancing_01/image_00105.jpg', 
                             'courtyard_dancing_01/image_00115.jpg', 'courtyard_dancing_01/image_00109.jpg', 
                             'courtyard_dancing_01/image_00112.jpg', 'courtyard_dancing_01/image_00110.jpg', 
                             'courtyard_dancing_01/image_00121.jpg', 'courtyard_dancing_01/image_00104.jpg', 
                             'courtyard_dancing_01/image_00103.jpg', 'courtyard_dancing_01/image_00122.jpg', 
                             'courtyard_hug_00/image_00177.jpg', 'courtyard_hug_00/image_00216.jpg', 
                             'courtyard_hug_00/image_00217.jpg']


def moving_average(data, window_size):
    """
    Calculate the moving average of a 1D list.

    Parameters:
    - data: The input list of numerical values.
    - window_size: The size of the moving average window.

    Returns:
    - A list containing the moving averages.
    """
    moving_averages = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        average = sum(window) / window_size
        moving_averages.append(average)
    return np.array(moving_averages).reshape(-1)

def running_mean(data):
    """
    Calculate the running mean of a 1D list.

    Parameters:
    - data: The input list of numerical values.

    Returns:
    - A list containing the running means.
    """
    running_means = []
    total_sum = 0

    for i, value in enumerate(data, start=1):
        total_sum += value
        mean = total_sum / i
        running_means.append(mean)

    return running_means

def running_sum(data):
    """
    Calculate the running mean of a 1D list.

    Parameters:
    - data: The input list of numerical values.

    Returns:
    - A list containing the running means.
    """
    running_means = []
    total_sum = 0

    for i, value in enumerate(data, start=1):
        total_sum += value
        mean = total_sum
        running_means.append(mean)

    return running_means

def plot_data(data, xtick_labels, labels):
    N, M = data.shape  # Assuming data is a NumPy array
    x = np.arange(M)

    plt.figure(figsize=(10, 8))  # Width, Height
    
    for i in range(N):
        plt.plot(x, data[i], label=labels[i])
    
    # Set custom X-axis tick labels
    plt.xticks(x, xtick_labels, rotation='vertical')
    
    # Add labels and legend
    plt.xlabel("Body Segments")
    plt.ylabel("Weight")
    plt.legend(title="k", loc='best')  # You can specify the legend location
    plt.tight_layout()
    
    # Show the plot
    plt.show()


def normalize_to_sum_one(float_list):
    # Calculate the sum of the list
    sum_of_list = sum(float_list)

    # Normalize the list so that the sum equals 1
    normalized_list = [x / sum_of_list for x in float_list]

    return normalized_list

def process_truncation_data(occlusion_anno_path, weights):
    seq_list = truncated_seqs

    regional_occlusion_list = []

    for seq_name in seq_list:

        #print("Processing {}...".format(seq_name))
        annotation_list = glob.glob('{}/{}/*.json'.format(occlusion_anno_path, seq_name))

        for frame_id, json_filename in enumerate(annotation_list):
            
            image_name = json_filename.split("/")[-1][:-5]
            image_path = "{}/{}.jpg".format(seq_name, image_name)

            if image_path in erroneous_annotation_list:
                continue

            with open('{}'.format(json_filename)) as json_filepointer:
                annotation_dict = json.load(json_filepointer)

            for model_id in list(annotation_dict.keys()):
                
                if model_id == "overall": # overall of the frame
                    continue

                occlusion_record = [image_path, model_id]
                occlusion_score = 0

                if annotation_dict[model_id]["truncation"]["overall"] != 0:
                    print(occlusion_record)
                    sys.exit()
                    # TODO truncation is calculated as NaN -> fix it by recalculating annotations

                for joint_id, joint_name in enumerate(annotation_dict[model_id]["truncation"].keys()):                      
                    body_segment_occlusion = annotation_dict[model_id]["truncation"][joint_name]
                    occlusion_score += weights[joint_id] * body_segment_occlusion

                
                occlusion_record.append(occlusion_score)
                regional_occlusion_list.append(occlusion_record)

    regional_occlusion_list = sorted(regional_occlusion_list, key=itemgetter(2), reverse=True)
    """
    with open("./3dpw/selected_frames/{}.txt".format(datetime.now()), "w+") as result_filepointer:
        for joint_id, joint_name in enumerate(selector):
            result_filepointer.write("{}:{}, ".format(joint_name, weights[joint_id]))

        result_filepointer.write("\n")

        for record in regional_occlusion_list:
            result_filepointer.write("{} {} {}\n".format(record[0], record[1], record[2]))
    """

    return regional_occlusion_list


def process_occlusion_data(occlusion_anno_path, category, weights, algorithm):
    seq_list = os.listdir(occlusion_anno_path)

    regional_occlusion_list = []

    for seq_name in seq_list:
        
        #print("Processing {}...".format(seq_name))
        annotation_list = glob.glob('{}/{}/*.json'.format(occlusion_anno_path, seq_name))

        for frame_id, json_filename in enumerate(annotation_list):
            
            image_name = json_filename.split("/")[-1][:-5]
            image_path = "{}/{}.png".format(seq_name, image_name) # removed _1280x720

            if image_path in erroneous_annotation_list:
                continue

            with open('{}'.format(json_filename)) as json_filepointer:
                annotation_dict = json.load(json_filepointer)

            for model_id in list(annotation_dict.keys()):
                
                if model_id == "overall":
                    continue

                occlusion_record = [image_path, model_id]
                occlusion_score = 0

                if algorithm == "regional_occlusion":
                    for joint_id, joint_name in enumerate(annotation_dict[model_id][category].keys()):                      
                        body_segment_occlusion = annotation_dict[model_id][category][joint_name]
                        body_segment_truncation = annotation_dict[model_id]["truncation"][joint_name]
                        if body_segment_truncation == 100:
                            continue
                        visible_region = 1-(body_segment_truncation/100)
                        occluded_region = visible_region * (body_segment_occlusion/100)
                        occlusion_score += weights[joint_id] * (occluded_region + (body_segment_truncation/100)) #((body_segment_occlusion/100) + (body_segment_truncation/100))
                elif algorithm == "occlusion_index":
                    for joint_id, joint_name in enumerate(annotation_dict[model_id]["occlusion"].keys()): 
                        body_segment_occlusion = annotation_dict[model_id]["occlusion"][joint_name]
                        body_segment_truncation = annotation_dict[model_id]["truncation"][joint_name]
                        if body_segment_truncation == 100:
                            continue
                        occlusion_score += weights[joint_id] * (float(body_segment_occlusion) + float(body_segment_truncation))
                elif algorithm == "crowd_index":
                    occlusion_score = annotation_dict[model_id]
                else:
                    print("Unknown algorithm")
                    sys.exit()
                
                occlusion_record.append(occlusion_score)
                regional_occlusion_list.append(occlusion_record)

    regional_occlusion_list = sorted(regional_occlusion_list, key=itemgetter(2), reverse=True)

    return regional_occlusion_list

def generate_weights(error_path, occlusion_path, k=10, category="occlusion", modified=False):
    with open(error_path, "r") as json_filepointer:
        error_dict = json.load(json_filepointer)

    occlusion_list = []

    for seq_name in list(error_dict.keys()):
        if seq_name in truncated_seqs:
            continue

        for frame_name in list(error_dict[seq_name].keys()):
            if "{}/{}".format(seq_name, frame_name) in erroneous_annotation_list:
                continue
            
            for model_id in list(error_dict[seq_name][frame_name].keys()):
                if model_id == "overall" or model_id == "n_preds": # overall of frame
                    continue
                
                if modified: # modified, error += joint segment visibility * joint_error 
                    with open('{}/{}/{}.json'.format(occlusion_path, seq_name, frame_name[:-13])) as json_filepointer:
                        annotation_dict = json.load(json_filepointer)

                    occlusion_rates = annotation_dict[model_id][category]
                    
                    error = 0
                    
                    joint_names = list(occlusion_rates.keys())

                    for joint_id in list(error_dict[seq_name][frame_name][model_id].keys())[1:]: # skip overall occlusion
                        error += (100-occlusion_rates[joint_names[int(joint_id)]]) * error_dict[seq_name][frame_name][model_id][joint_id]
                else:
                    error = error_dict[seq_name][frame_name][model_id]["overall"] # overall of joints with equal weighting

                occlusion_list.append([seq_name, frame_name, model_id, error]) 

    occlusion_list = sorted(occlusion_list, key=itemgetter(3), reverse=True)

    bad_annotation_list = []

    occlusion_ratios = []   
    i = 0
    for occlusion_anno in occlusion_list[:k]:
        [seq_name, frame_name, model_id, error] = occlusion_anno

        with open('{}/{}/{}.json'.format(occlusion_path, seq_name, frame_name[:-4])) as json_filepointer:
            annotation_dict = json.load(json_filepointer)
        
        occlusion_rates = annotation_dict[model_id][category]
        occlusion_ratios.append(list(occlusion_rates.values()))
        
        overall_occlusion_last = occlusion_ratios[-1][0]
        """
        if overall_occlusion_last < 10:

            img = cv2.imread('{}/{}/{}'.format(occlusion_path, seq_name, frame_name))

            img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))

            cv2.imshow("{}/{}".format(seq_name, frame_name), img)

            k = cv2.waitKey(0)
            if k==32:    # Esc key to stop
                print(i, seq_name, frame_name, model_id, occlusion_ratios[-1])
                bad_annotation_list.append("{}/{}".format(seq_name, frame_name))
            else:
                pass

            cv2.destroyAllWindows()
        """
            
        i += 1
    
    #print(bad_annotation_list)
    return np.mean(occlusion_ratios, axis=0), occlusion_rates.keys()


def calculate_error(error_path, occlusion_path, occlusion_list, k=10, error_algorithm="standard", drop_zero_occ = False):

    with open(error_path, "r") as json_filepointer:
        error_dict = json.load(json_filepointer)

    errors = []

    occlusion_scores = []

    prediction_missing = []

    for occlusion_record in occlusion_list[:k]:
        [image_path, model_id, occlusion_score] = occlusion_record
        [seq_name, frame_name] = image_path.split("/")

        if drop_zero_occ and occlusion_score == 0:
            continue

        occlusion_scores.append(occlusion_score)
        
        error_values = list(error_dict[seq_name][frame_name][model_id].values())
        if error_values[0] == 9999:
            
            prediction_missing.append(1)
            if "pa_mpjpe" in error_path:
                errors.append([1.0] * 25)
            else:
                errors.append([5.0] * 25)
            continue

        prediction_missing.append(0)

        if error_algorithm == "standard": 
            errors.append(error_values)

        elif error_algorithm == "keypoint_visibility":
            with open('{}roi_json/{}/{}.json'.format(occlusion_path, seq_name, frame_name[:-4])) as json_filepointer:
                annotation_dict = json.load(json_filepointer)
            occlusion_values = annotation_dict[model_id]["occlusion"]

            for joint_id, joint_name in enumerate(occlusion_values.keys()):
                if type(occlusion_values[joint_name]) == float:
                    visibility_value_coeff = 1 - occlusion_values[joint_name]
                else:
                    visibility_value_coeff = int(~occlusion_values[joint_name])
                error_values[joint_id] *= visibility_value_coeff

            errors.append(error_values)

        elif error_algorithm == "truncation":
            with open('{}roi_json/{}/{}.json'.format(occlusion_path, seq_name, frame_name[:-4])) as json_filepointer:
                annotation_dict = json.load(json_filepointer)

            truncation_values = annotation_dict[model_id]["truncation"]

            for joint_id, joint_name in enumerate(truncation_values.keys()):
                if type(truncation_values[joint_name]) == float:
                    visibility_value_coeff = 1 - truncation_values[joint_name]
                else:
                    visibility_value_coeff = int(~truncation_values[joint_name])
                error_values[joint_id] *= visibility_value_coeff

            errors.append(error_values)

        elif error_algorithm == "region_visibility":
            with open('{}roi_json/{}/{}.json'.format(occlusion_path, seq_name, frame_name[:-4])) as json_filepointer:
                annotation_dict = json.load(json_filepointer)

            regional_occlusion_values = annotation_dict[model_id]["occlusion"]

            for joint_id, joint_name in enumerate(regional_occlusion_values.keys()):   
                visibility_value_coeff = (100 - regional_occlusion_values[joint_name]) / 100

                error_values[joint_id] *= visibility_value_coeff

            errors.append(error_values)

        else:
            print("Unknown error_algorithm")
            sys.exit()

        #print(occlusion_score, errors[-1][0])
        #print(image_path, model_id, )

        
    return errors, occlusion_scores, prediction_missing

def drop_erroneous(error_path, occlusion_root_path):
    with open(error_path, "r") as json_filepointer:
        error_dict = json.load(json_filepointer)

    occlusion_list = []

    for seq_name in list(error_dict.keys()):
        if seq_name in truncated_seqs:
            continue

        for frame_name in list(error_dict[seq_name].keys()):
            if "{}/{}".format(seq_name, frame_name) in erroneous_annotation_list:
                continue
            
            for model_id in list(error_dict[seq_name][frame_name].keys()):
                if model_id == "overall" or model_id == "n_preds": # overall of frame
                    continue

                error = error_dict[seq_name][frame_name][model_id]["overall"] # overall of joints with equal weighting

                occlusion_list.append([seq_name, frame_name, model_id, error]) 

    occlusion_list = sorted(occlusion_list, key=itemgetter(3), reverse=True)

    regional_path = occlusion_root_path + "regional_occlusion"
    oi_path = occlusion_root_path + "occlusion_index"
    ci_path = occlusion_root_path + "crowd_index"

    bad_annotation_list = []

    for occlusion_anno in occlusion_list:
        [seq_name, frame_name, model_id, error] = occlusion_anno

        with open('{}/{}/{}.json'.format(regional_path, seq_name, frame_name[:-4])) as json_filepointer:
            regional_annotation_dict = json.load(json_filepointer)

            roi = regional_annotation_dict[model_id]["occlusion"]["overall"]

        with open('{}/{}/{}.json'.format(oi_path, seq_name, frame_name[:-4])) as json_filepointer:
            oi_annotation_dict = json.load(json_filepointer)

            oi = oi_annotation_dict[model_id]["occlusion"]["overall"]

        with open('{}/{}/{}.json'.format(ci_path, seq_name, frame_name[:-4])) as json_filepointer:
            ci_annotation_dict = json.load(json_filepointer)

            ci = ci_annotation_dict[model_id]
        
        img = cv2.imread('{}/{}/{}'.format("/home/tuba/Documents/emre/thesis/occlusionIndex/3dpw/masks", seq_name, frame_name))
        mask = cv2.imread('{}/{}/{}'.format("/home/tuba/Documents/emre/thesis/occlusionIndex/3dpw/occlusion_index", seq_name, frame_name))

        result = cv2.imread('{}/{}/{}'.format("/home/tuba/Documents/emre/thesis/occlusionIndex/methods/romp_results/3dpw/", seq_name, frame_name))

        img = cv2.resize(img, (int(img.shape[1] * 0.25), int(img.shape[0] * 0.25)))
        mask = cv2.resize(mask, (int(mask.shape[1] * 0.25), int(mask.shape[0] * 0.25)))
        result = cv2.resize(result, (int(result.shape[1] * 0.25), int(result.shape[0] * 0.25)))

        print("{}/{}".format(seq_name, frame_name))
        concat_img = np.concatenate(tuple([img, mask, result]), axis=1)

        cv2.imshow("RIO: {:.2f} | OI: {:.2f} | CI: {:.2f} | Error: {}".format(roi, oi, ci, error), concat_img)

        k = cv2.waitKey(0)
        if k==32:    # space key to stop
            #print(i, seq_name, frame_name, model_id, occlusion_ratios[-1])
            bad_annotation_list.append("{}/{}".format(seq_name, frame_name))
            pass
        else:
            pass

        cv2.destroyAllWindows()
            
    print(bad_annotation_list)
    #return np.mean(occlusion_ratios, axis=0), occlusion_rates.keys()

def closest_elements_indices(lists, target):
    """
    Find the indices of the elements in each list that are closest to the target number.

    Parameters:
    - lists: A list of lists containing numeric elements.
    - target: The target number.

    Returns:
    A list of tuples where each tuple contains the indices of the closest elements in each list.
    """
    closest_indices = []

    for lst in lists:
        closest_index = min(range(len(lst)), key=lambda i: abs(lst[i] - target))
        closest_indices.append(closest_index)

    return closest_indices
