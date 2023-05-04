import os

import cv2
import numpy as np

def read_img(img_path, scale=1, show=False):
	img = cv2.imread(img_path)

	img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

	if show:
		show_img(img)

	return img

def show_img(img, frame_id):
	cv2.imshow(str(frame_id), img)

	cv2.waitKey(0)

	cv2.destroyAllWindows()


def save_processed_img(img, img_filename):
	os.makedirs("processed_imgs", exist_ok=True)

	cv2.imwrite("processed_imgs/{}".format(img_filename), img)


def draw_keypoints(image, keypoints, colors, occluded_keypoints = None):

	for model_id in range(len(keypoints)):

		keypoints_converted = keypoint_type_conversion(keypoints[model_id])


		if occluded_keypoints != None: # if occluded provided, then only draw them
			new_keypoints_converted = []
			for id in occluded_keypoints[model_id]:
				new_keypoints_converted.append(keypoints_converted[id])

			keypoints_converted = new_keypoints_converted



		image = cv2.drawKeypoints(image, keypoints_converted, 0, colors[model_id])

	return image


def draw_masks(image, masks, colors):

	num_models = len(masks)

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

	for model_id in range(num_models):

		if colors == None:
			color = np.random.randint(255, size=3).tolist()


		left_top = bboxes[model_id][0]

		right_bottom = bboxes[model_id][1]

		image = cv2.rectangle(image, left_top, right_bottom, colors[model_id], 2)
		image = cv2.putText(image, "{:.2f}".format(index_values[model_id]), left_top, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

	return image


def keypoint_type_conversion(keypoints, scale=1, keypoint_size=100):

	num_keypoints = len(keypoints)

	keypoints_list = []

	for i in range(num_keypoints):

		converted_keypoint = cv2.KeyPoint(float(keypoints[i][0]*scale), float(keypoints[i][1]*scale), keypoint_size)

		keypoints_list.append(converted_keypoint)

	keypoints_list = tuple(keypoints_list)


	return keypoints_list




def draw_sidebyside(img1, img2, vis_scale= 1, show=True):

	pad = 0

	img1 = cv2.copyMakeBorder(img1, pad, pad, pad, pad, cv2.BORDER_CONSTANT)

	img2 = cv2.copyMakeBorder(img2, pad, pad, pad, pad, cv2.BORDER_CONSTANT)

	numpy_horizontal_concat = np.concatenate((img1, img2), axis=1)

	if show:
		numpy_horizontal_concat = cv2.resize(numpy_horizontal_concat, (int(numpy_horizontal_concat.shape[1] * vis_scale), int(numpy_horizontal_concat.shape[0] * vis_scale)))

		cv2.imshow("Side by Side", numpy_horizontal_concat)

		cv2.waitKey(0)

		cv2.destroyAllWindows()

	return numpy_horizontal_concat


# NOT USED

def draw_bbox(img, keypoints, color=None, occlusion_ratio=None):
	if color==None:
		color = np.random.randint(255, size=3).tolist()
	x_offset = 20
	y_offset = 20

	nonzero_keypoints = keypoints[np.nonzero(keypoints)[0]]


	left_top = [int(nonzero_keypoints[:, 0].min() - x_offset), int(nonzero_keypoints[:, 1].min() - y_offset)]

	right_bottom = [int(keypoints[:, 0].max() + x_offset), int(keypoints[:, 1].max() + y_offset)]

	bbox_img = cv2.rectangle(img, left_top, right_bottom, color, 2)
	bbox_img = cv2.putText(bbox_img, "{:.2f}".format(occlusion_ratio),
						   left_top, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
						   (255, 255, 255), 2)

	return bbox_img

def draw_mask_keypoint(image, masks, keypoints, bbox, occlusion_indices, occluded_keypoints=None):

	num_models = len(keypoints)
	colors = np.random.randint(255, size=(num_models, 3)).tolist()# each person has its own color

	for model_id in range(len(masks)):

		for mask in masks[model_id]: # occlusion leads more than one piece of mask
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

	for model_id in range(len(keypoints)):

		keypoints_converted = keypoint_type_conversion(keypoints[model_id])

		if occluded_keypoints != None:
			new_keypoints_converted = []
			for id in occluded_keypoints[model_id]:

				new_keypoints_converted.append(keypoints_converted[id])

			keypoints_converted = new_keypoints_converted

		image = cv2.drawKeypoints(image, keypoints_converted, 0, colors[model_id])

		image = draw_bbox(image, keypoints=keypoints[model_id], color=colors[model_id], occlusion_ratio = occlusion_indices[model_id])

	return image
