import numpy as np
import cv2
import os, json


def show_img(img, window_name):
	cv2.imshow(str(window_name), img)

	cv2.waitKey(0)

	cv2.destroyAllWindows()


def save_processed_img(img, img_filename):
	os.makedirs("processed_imgs", exist_ok=True)

	cv2.imwrite("processed_imgs/{}".format(img_filename), img)

def draw_keypoints(image, keypoints, occlusion_status, colors, draw_red=False):
	num_models = len(keypoints)

	if colors == None:
		colors = np.random.randint(255, size=(num_models, 3)).tolist()  # try not to send colors None

	for model_id in range(len(keypoints)):

		for kp_id, kp in enumerate(keypoints[model_id]):
			if occlusion_status[model_id][kp_id]:
				#continue
				if draw_red:
					image = cv2.circle(image, (int(kp[0]), int(kp[1])), 2, (0, 0, 255), 5)
				else:
					image = cv2.circle(image, (int(kp[0]), int(kp[1])), 2, colors[model_id], 5)
			#image = cv2.putText(image, "{}".format(occlusion_status[model_id, kp_id]), (int(kp[0]), int(kp[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


	return image


def draw_masks(image, masks, colors):

	num_models = len(masks)

	if colors == None:
		colors = np.random.randint(255, size=(num_models, 3)).tolist()  # try not to send colors None

	for model_id in range(num_models):

		for mask in masks[model_id]:  # occlusion leads more than one piece of mask
			# Polygon corner points coordinates
			pts = np.array(mask,
						   np.int32)

			pts = pts.reshape((-1, 1, 2))

			isClosed = True

			# Line thickness of 2 px
			thickness = 2

			alpha = 0.6

			# Using cv2.polylines() method
			# Draw a Blue polygon with
			# thickness of 1 px
			overlay = image.copy()
			cv2.fillPoly(overlay, [pts], colors[model_id])
			image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

	return image

def draw_bboxes(image, bboxes, colors, index_values):

	num_models = len(bboxes)

	if colors == None:
		colors = np.random.randint(255, size=(num_models, 3)).tolist()  # try not to send colors None

	for model_id in range(num_models):

		left_top = bboxes[model_id][0]

		right_bottom = bboxes[model_id][1]

		image = cv2.rectangle(image, left_top, right_bottom, colors[model_id], 2)
		if index_values != None:
			image = cv2.putText(image, "{:.2f}".format(index_values[model_id]), left_top, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

	return image


def draw_sidebyside(img1, img2, vis_scale= 1, show=True, title="Side by Side"):

	pad = 0

	img1 = cv2.copyMakeBorder(img1, pad, pad, pad, pad, cv2.BORDER_CONSTANT)

	img2 = cv2.copyMakeBorder(img2, pad, pad, pad, pad, cv2.BORDER_CONSTANT)

	numpy_horizontal_concat = np.concatenate((img1, img2), axis=1)

	if show:
		numpy_horizontal_concat = cv2.resize(numpy_horizontal_concat, (int(numpy_horizontal_concat.shape[1] * vis_scale), int(numpy_horizontal_concat.shape[0] * vis_scale)))

		show_img(numpy_horizontal_concat, title)

	return numpy_horizontal_concat