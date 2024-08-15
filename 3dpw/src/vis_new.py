import cv2, os
import numpy as np

color_list = [
    [255, 0, 0], # red
    [0, 0, 255], # blue
    [179, 179, 179], # pink
    [230, 230, 230], # neutral
    [179, 192, 128], # capsule
    [128, 179, 192] # yellow
]

def show_im(img):

    cv2.imshow("window_name", img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

def read_im(img_path, scale=1, show=False):
    img = cv2.imread(img_path)

    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

    if show:
        cv2.imshow("window_name", img)

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    return img

def save_processed_img(img, img_filename, scene_name="courtyard_basketball_00", folder_path="./3dpw/processed_imgs"):
	os.makedirs(folder_path + "/" + scene_name, exist_ok=True)

	cv2.imwrite("{}/{}/{}".format(folder_path, scene_name, img_filename), img)

def draw_bboxes(image, bboxes, colors, index_values):

    num_models = len(bboxes)
    
    if colors == None:
        colors = np.random.randint(255, size=(num_models, 3)).tolist()  # try not to send colors None

    for model_id in range(num_models):

        left_top = bboxes[model_id][0]

        right_bottom = bboxes[model_id][1]

        image = cv2.rectangle(image, left_top, right_bottom, colors[model_id], 2)

        if index_values != None:            
            image = cv2.putText(image, "{:.2f}".format(index_values[model_id]), left_top, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    return image


def draw_keypoints(image, keypoints, occlusion_index, kp_names, only_occluded=False, colors=None):
    num_models = len(keypoints)

    if colors == None:
        colors = (np.ones((num_models, 3)) * 255 ).tolist()  # try not to send colors None

    for model_id in range(num_models):
        for kp_id, kp in enumerate(keypoints[model_id]):

            kp_name = kp_names[kp_id]
            
            if occlusion_index[model_id]["truncation"][kp_name]:
                continue

            if only_occluded:
                if occlusion_index[model_id]["occlusion_index"][kp_name]:
                    image = cv2.circle(image, (int(kp[0]), int(kp[1])), 1, colors[model_id], 2)
                else:
                    pass
            else:
                image = cv2.circle(image, (int(kp[0]), int(kp[1])), 3, colors[model_id], 7)
    return image

def draw_masks(image, mask):

    mask[mask == (0, 0, 255)] = 0 

    alpha = 0.3

    # Using cv2.polylines() method
    # Draw a Blue polygon with
    # thickness of 1 px

    image = cv2.addWeighted(mask, alpha, image, 1 - alpha, 0)

    return image