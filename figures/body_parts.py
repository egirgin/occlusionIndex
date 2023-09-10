import os
import sys
import json
import trimesh
import subprocess
import numpy as np
from smplx import SMPL, SMPLH, SMPLX
from matplotlib import cm as mpl_cm, colors as mpl_colors
from smpl_np_romp import SMPLModel


def download_url(url, outdir):
    print(f'Downloading files from {url}')
    cmd = ['wget', '-c', url, '-P', outdir]
    subprocess.call(cmd)
    file_path = os.path.join(outdir, url.split('/')[-1])
    return file_path


def part_segm_to_vertex_colors(part_segm, n_vertices, alpha=1.0):
    vertex_labels = np.zeros((n_vertices, 3))
    #vertex_labels = np.zeros(n_vertices)
    """
    rightHand rightUpLeg leftArm leftLeg leftToeBase leftFoot spine1 spine2 leftShoulder rightShoulder
    rightFoot head rightArm leftHandIndex1 rightLeg rightHandIndex1 leftForeArm rightForeArm neck
    rightToeBase spine leftUpLeg leftHand hips
    """
    light_blue = [51, 153, 255]
    light_pink = [255, 51, 153]
    light_green = [153, 255, 102]
    light_orange = [255, 153, 51]
    magenta = [204, 102, 255]
    turquoise = [0, 255, 255]
    grey = [102, 153, 153]

    for part_idx, (k, v) in enumerate(part_segm.items()):
        print(k)
        if k in ["head", "neck"]:
            vertex_labels[v] = grey
        elif k in ["rightHand", "rightShoulder", "rightArm", "rightHandIndex1", "rightForeArm"]:
            vertex_labels[v] = light_pink
        elif k in ["leftArm", "leftShoulder", "leftHandIndex1", "leftForeArm", "leftHand"]:
            vertex_labels[v] = turquoise
        elif k in ["rightUpLeg", "rightFoot", "rightLeg", "rightToeBase"]:
            vertex_labels[v] = light_blue
        elif k in ["leftLeg", "leftToeBase", "leftFoot", "leftUpLeg"]:
            vertex_labels[v] = magenta
        elif k in ["spine", "hips"]:
            vertex_labels[v] = light_orange
        elif k in ["spine1", "spine2"]:
            vertex_labels[v] = light_green
        else:
            vertex_labels[v] = 7#part_idx

    cm = mpl_cm.get_cmap('jet')
    norm_gt = mpl_colors.Normalize()

    vertex_colors = np.ones((n_vertices, 4))
    vertex_colors[:, 3] = alpha
    vertex_colors[:, :3] = vertex_labels/255. #cm(norm_gt(vertex_labels))[:, :3]
    #print(vertex_colors[0, :3])
    return vertex_colors


def main():
    main_url = 'https://raw.githubusercontent.com/Meshcapade/wiki/main/assets/SMPL_body_segmentation/'

    part_segm_url = os.path.join(main_url, 'smpl/smpl_vert_segmentation.json')
    body_model = SMPLModel('/home/tuba/Documents/emre/thesis/models/converted/SMPL_MALE.pkl')

    part_segm_filepath = download_url(part_segm_url, '.')
    part_segm = json.load(open(part_segm_filepath))

    beta = np.zeros(10)  # (np.random.rand(*smpl.beta_shape) - 0.5) * 0.06
    
    pose = np.zeros((24,3)) # (np.random.rand(*smpl.pose_shape) - 0.5) * 0.4
    trans = np.zeros(3)  # np.zeros(smpl.trans_shape)

    vertices = body_model.set_params(beta=beta, pose=pose, trans=trans)

    faces = body_model.faces

    vertex_colors = part_segm_to_vertex_colors(part_segm, vertices.shape[0])

    mesh = trimesh.Trimesh(vertices, faces, process=False, vertex_colors=vertex_colors)
    mesh.show(background=(0,0,0,0))

    


if __name__ == '__main__':
    main()