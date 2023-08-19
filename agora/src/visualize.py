import cv2, os
import numpy as np

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

def save_processed_img(img, img_filename):
	os.makedirs("processed_imgs", exist_ok=True)

	cv2.imwrite("processed_imgs/{}".format(img_filename), img)

def draw_bboxes(image, bboxes, colors, index_values):

    num_models = len(bboxes)

    if colors == None:
        colors = np.random.randint(255, size=(num_models, 3)).tolist()  # try not to send colors None

    for model_id in range(num_models):

        left_top = bboxes[model_id][0]

        right_bottom = bboxes[model_id][1]

        image = cv2.rectangle(image, left_top, right_bottom, colors[model_id], 2)
        if index_values != None:
            image = cv2.putText(image, "{:.2f}".format(index_values[model_id]), left_top, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image


def draw_keypoints(image, keypoints, occlusion_status, only_occluded=False):
    num_models = len(keypoints)

    for model_id in range(num_models):
        for kp_id, kp in enumerate(keypoints[model_id]):
            if only_occluded and occlusion_status[model_id][kp_id]:
                image = cv2.circle(image, (int(kp[0]), int(kp[1])), 1, (0, 0, 255), 2)
            else:
                image = cv2.circle(image, (int(kp[0]), int(kp[1])), 1, (0, 0, 255), 2)
            #image = cv2.putText(image, "{}".format(occlusion_status[model_id, kp_id]), (int(kp[0]), int(kp[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return image

def draw_masks(image, mask):

    mask[mask == (0, 0, 255)] = 0 

    alpha = 0.3

    # Using cv2.polylines() method
    # Draw a Blue polygon with
    # thickness of 1 px

    image = cv2.addWeighted(mask, alpha, image, 1 - alpha, 0)

    return image