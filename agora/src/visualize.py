import cv2
import numpy as np

def approximate_bb(pose2d, scale):
    x_offset = 7
    y_offset = 15
    xs = pose2d[0]
    ys = pose2d[1]

    x_min = int(xs.min()/scale) - x_offset
    x_max = int(xs.max()/scale) + x_offset

    y_min = int(ys.min()/scale) - y_offset
    y_max = int(ys.max()/scale) + y_offset

    top_left = [x_min, y_min]

    bottom_right = [x_max, y_max]

    return top_left, bottom_right

def draw_sidebyside(img1, img2, vis_scale= 1, show=True):

    pad = 10

    img1 = cv2.copyMakeBorder(img1, pad, pad, pad, pad, cv2.BORDER_CONSTANT)

    img2 = cv2.copyMakeBorder(img2, pad, pad, pad, pad, cv2.BORDER_CONSTANT)

    numpy_horizontal_concat = np.concatenate((img1, img2), axis=1)

    if show:
        numpy_horizontal_concat = cv2.resize(numpy_horizontal_concat, (int(numpy_horizontal_concat.shape[1] * vis_scale), int(numpy_horizontal_concat.shape[0] * vis_scale)))

        cv2.imshow("Side by Side", numpy_horizontal_concat)

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    return numpy_horizontal_concat

def keypoint_type_conversion(keypoints, scale, keypoint_size=10):

    num_ppl, num_keypoints, _ = keypoints.shape

    keypoints_list = []

    for person_id in range(num_ppl):
        person_keypoints = [cv2.KeyPoint(keypoints[person_id][i][0]*scale, keypoints[person_id][i][1]*scale, keypoint_size) for i in range(num_keypoints)]

        person_keypoints = tuple(person_keypoints)

        keypoints_list.append(person_keypoints)

    return keypoints_list


def draw_keypoints_show(keypoint_img, keypoints2d, bboxes, vis_scale, show=True):

    ppl, _, _ = keypoints2d.shape

    for model_id in range(ppl):
        color = np.random.randint(0, 255, 3).tolist()

        keypoints_converted = keypoint_type_conversion(keypoints2d, vis_scale)

        keypoint_img = cv2.drawKeypoints(keypoint_img, keypoints_converted[model_id], 0, color)
        keypoint_img = cv2.rectangle(keypoint_img, np.multiply(bboxes[model_id][0], vis_scale).astype(int), np.multiply(bboxes[model_id][1], vis_scale).astype(int), color, 2)

    if show:

        cv2.imshow("BBoxes and 2D Keypoints", keypoint_img)

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    return keypoint_img


def draw_keypoints_mask(all_mask, keypoints2d, scale, show=True, count=0):
    ppl, _, _ = keypoints2d.shape

    for model_id in range(ppl):
        keypoints_converted = keypoint_type_conversion(keypoints2d, scale)

        all_mask = cv2.drawKeypoints(all_mask, keypoints_converted[model_id], 0, (255,255,255))

    if show:
        cv2.imshow("Mask and 2D Keypoints {}".format(count), all_mask)

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    return all_mask


def draw_crowd_keypoints(img, keypoints, bboxes, crowd_indices, scale, show=True):

    for model_id in range(len(bboxes)):
        color = np.random.randint(0, 255, 3).tolist()
        keypoint_img = cv2.rectangle(img, np.multiply(bboxes[model_id][0], scale).astype(int), np.multiply(bboxes[model_id][1], scale).astype(int), color, 2)
        #keypoint_img = cv2.putText(keypoint_img, "{:.2f}".format(crowd_indices[model_id]), np.multiply(bboxes[model_id][0], scale).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        try:
            keypoint_img = cv2.putText(keypoint_img, "{:.2f}".format(crowd_indices[model_id]),
                                   np.multiply(bboxes[model_id][0], scale).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                   (0, 0, 0), 2)
        except:
            pass

    keypoints_converted = keypoint_type_conversion(keypoints, scale)

    keypoint_img = cv2.drawKeypoints(keypoint_img, keypoints_converted[0], 0, np.random.randint(0, 255, 3).tolist())

    if show:
        cv2.imshow("Intersection Keypoints on Bboxes", keypoint_img)

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    return keypoint_img


def draw_mask_crowd_keypoints(mask_img, keypoints, bboxes, keypoint_occlusion_ratio, scale, show=True):

    for model_id in range(len(bboxes)):
        color = np.random.randint(0, 255, 3).tolist()
        #mask_img = cv2.rectangle(mask_img, np.multiply(bboxes[model_id][0], scale).astype(int), np.multiply(bboxes[model_id][1], scale).astype(int), color, 2)
        #mask_img = cv2.putText(mask_img, "{:.2f}".format(keypoint_occlusion_ratio[model_id]), np.multiply(bboxes[model_id][0], scale).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        try:
            mask_img = cv2.putText(mask_img, "{:.2f}".format(keypoint_occlusion_ratio[model_id]),
                               np.multiply(bboxes[model_id][0], scale).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                               (255, 255, 255), 2)
        except:
            pass
    keypoints_converted = keypoint_type_conversion(keypoints, scale)

    mask_img = cv2.drawKeypoints(mask_img, keypoints_converted[0], 0, [255,255,255])

    if show:
        cv2.imshow("Intersection Keypoints on Masks", mask_img)

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    return mask_img
