import trimesh
import numpy as np
import cv2
import os
import json

global image_width, image_height, image_canvas, z_buffer

# Create an image canvas
image_width = 1080 # adjust # TODO
image_height = 1920
image_canvas = np.zeros((image_height, image_width, 3), dtype=np.uint8)

# Create a Z-buffer to store depth values
z_buffer = np.ones((image_height, image_width), dtype=np.float64) * -np.inf

color_list = [
    [255, 0, 0], # red
    [0, 0, 255], # blue
    [.7, .7, .9] * 255, # pink
    [.9, .9, .8] * 255, # neutral
    [.7, .75, .5] * 255, # capsule
    [.5, .7, .75] * 255 # yellow
]


def load_mesh(path_list):
    vertices_list = []
    faces_list = []

    for path in path_list:
        mesh_obj = trimesh.load_mesh(path)
        vertices_list.append(mesh_obj.vertices)
        faces_list.append(mesh_obj.faces)
    
    return vertices_list, faces_list

def load_cam(cam_path):

    with open(cam_path, "r") as cam_json:
        cam_data = json.load(cam_json)

    intrinsics = np.array(cam_data["intrinsics"])
    extrinsics = np.array(cam_data["extrinsics"])

    return intrinsics, extrinsics

def project_points(vertices_list, intrinsics, extrinsics):
    image_points_list = []

    for vertices in vertices_list:
        rotation_matrix = extrinsics[:3, :3]  # 3x3 rotation matrix
        translation_vector = extrinsics[:3, -1]  # 3x1 translation vector

        projection_matrix = np.dot(intrinsics, np.hstack((rotation_matrix, translation_vector.reshape(-1, 1))))
        projected_vertices = np.dot(projection_matrix, np.hstack((vertices, np.ones((len(vertices), 1)))).T).T # move to homogeneous coords
        image_points = projected_vertices[:, :2] / projected_vertices[:, 2].reshape(-1, 1) # get back from homogeneous coords.
        image_points_list.append(image_points)

    return image_points_list

def render(faces, image_points, vertices, color):
    # Draw triangles on the image canvas with proper depth testing
    for face in faces:
        triangle = np.array([image_points[face[0]], image_points[face[1]], image_points[face[2]]], dtype=np.int32)
        
        # Calculate the bounding box of the triangle
        min_x = max(min(triangle[:, 0]), 0)
        max_x = min(max(triangle[:, 0]), image_width - 1)
        min_y = max(min(triangle[:, 1]), 0)
        max_y = min(max(triangle[:, 1]), image_height - 1)
        
        # Iterate over the bounding box and perform depth testing
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):

                inside_test = cv2.pointPolygonTest(triangle, (int(x), int(y)), measureDist=True)

                if inside_test >= 0:
                    barycentric_triangle = np.hstack((triangle, np.ones((len(triangle), 1)))).T # move to homogeneous coords and convert to form below
                    # ax bx cx
                    # ay by cy
                    #  1  1  1
                    
                    # calculate barycentric coords
                    (alpha, beta, gamma) = np.linalg.lstsq(barycentric_triangle, (int(x), int(y), 1), rcond=None)[0] # point in homogeneous coords

                    depth = vertices[face[0], 2] * alpha + \
                            vertices[face[1], 2] * beta + \
                            vertices[face[2], 2] * gamma
                    
                    
                    if depth > z_buffer[y, x]:
                        z_buffer[y, x] = depth
                        image_canvas[y, x] = color  # Set pixel color

def main():

    smpl_obj_path = "/home/tuba/Documents/emre/thesis/occlusionIndex/3dpw/smpl_objects"

    for seq_name in os.listdir(smpl_obj_path):
        print("Processing {}...".format(seq_name))
        
        seq_path = smpl_obj_path + "/" + seq_name

        image_list = os.listdir(seq_path)
        image_list.sort()
        #image_list = ["image_00139"]:cheating to process only one image

        for image_name in image_list:
            print(image_name)
            # initialize canvas and z-buffer
            global image_canvas
            image_canvas = np.zeros((image_height, image_width, 3), dtype=np.uint8)

            global z_buffer
            z_buffer = np.ones((image_height, image_width), dtype=np.float64) * -np.inf
            
            # collect mesh paths
            image_folder_path = seq_path + "/" + image_name
            
            mesh_path_list = []

            cam_path = None

            for mesh_file in os.listdir(image_folder_path):
                if mesh_file.endswith(".obj"):
                    mesh_path_list.append(image_folder_path + "/" + mesh_file)
                else:
                    cam_path = image_folder_path + "/" + mesh_file

            # load mesh object
            vertices_list, faces_list = load_mesh(mesh_path_list)

            # load cam params
            intrinsics, extrinsics = load_cam(cam_path)

            # project triangles to image plane
            image_points_list = project_points(vertices_list=vertices_list, intrinsics=intrinsics, extrinsics=extrinsics)

            # render meshes with z buffer
            for mesh_id, _ in enumerate(mesh_path_list):
                render(faces=faces_list[mesh_id], image_points=image_points_list[mesh_id], vertices=vertices_list[mesh_id], color=color_list[mesh_id])

            # save generated mask
            save_path = "./3dpw/masks/{}".format(seq_name)

            os.makedirs(save_path, exist_ok=True)

            cv2.imwrite("{}/{}.jpg".format(save_path, image_name), image_canvas)
            
            # show results
            # cv2.imshow('result', image_canvas)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()