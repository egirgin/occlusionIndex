from dataloader_hps import HPS_Dataloader, Dataset
from vis import draw_keypoints, draw_masks, draw_sidebyside, draw_bboxes, save_processed_img
from utils import COCO_JOINT, joint_list, COCO_SUBSET, form_criterion
import numpy as np

def draw(record):
    img = record.img()

    # Draw occlusion image
    occlusion_img = draw_keypoints(img, keypoints=record.keypoints2d, occlusion_status=record.occlusion_status, colors=record.instance_colors, draw_red=True)
    occlusion_img = draw_masks(occlusion_img, masks=record.instance_masks, colors=record.instance_colors)
    occlusion_img = draw_bboxes(occlusion_img, bboxes=record.bboxes, colors=record.instance_colors, index_values=record.occlusion_indices) # TODO

    # Draw crowd image
    crowd_img = draw_keypoints(img, keypoints=record.keypoints2d, occlusion_status=record.occlusion_status, colors=record.instance_colors, draw_red=True)
    crowd_img = draw_bboxes(crowd_img, bboxes=record.bboxes, colors=record.instance_colors, index_values=None) # TODO

    concat_img = draw_sidebyside(img1=crowd_img, img2=occlusion_img, vis_scale=1, show=True)
    #save_processed_img(concat_img, img_path)


if __name__ == '__main__':

    corrupt_counter = 0
    
    missing_counter = [[0] * 19]
    occluded_counter = [[0] * 19]

    ochuman_dataset = HPS_Dataloader(dataset=Dataset.OCHuman)
    
    #draw(ochuman_dataset[2])
    
    for sample_id in range(ochuman_dataset.num_samples):
        record = ochuman_dataset[sample_id]
        """
        if record == None:
            corrupt_counter += 1
        else:
            miss = np.sum(record.missing_status, axis=0).tolist()
            missing_counter += [miss]

            occ = np.sum(record.occlusion_status, axis=0).tolist()
            occluded_counter += [occ]
        """
    """
    missing_counter = np.array(missing_counter)
    occluded_counter = np.array(occluded_counter)



    missing_density = np.sum(missing_counter, axis=0)
    occlusion_density = np.sum(occluded_counter, axis=0)

    for i in range(19):
        print("{}:{}".format(joint_list[i], occlusion_density[i]))

    print(corrupt_counter)
    """
    torso_criterion = form_criterion(COCO_SUBSET.torso_subset)

    torso_samples = ochuman_dataset.sort_samples(criterion=torso_criterion, drop_zeros=True) # check this, probably causing error

    for sample in torso_samples:
        draw(ochuman_dataset[sample])

    #ochuman_dataset[1].img(show=True)
    

    