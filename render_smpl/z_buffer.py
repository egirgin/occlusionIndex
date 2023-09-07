import trimesh
import numpy as np
import cv2

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
    # Define camera intrinsics
    intrinsics = [[1.96185286e+03, 0.00000000e+00, 5.40000000e+02],
                [0.00000000e+00, 1.96923077e+03, 9.60000000e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]


    # Define camera extrinsics
    extrinsics = np.array(
        [[ 0.89695426,  0.0636676 , -0.43751513, -1.05262216],
        [ 0.1627256 , -0.96765269,  0.19279172,  0.66311684],
        [-0.41108811, -0.24412027, -0.87830055,  2.78461675],
        [ 0.        ,  0.        ,  0.        ,  1.        ]]
        )

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

    mesh_path_list = [
        "/home/emre/Documents/master/thesis/custom_smpl/mesh_models/smpl_np_0.obj",
        "/home/emre/Documents/master/thesis/custom_smpl/mesh_models/smpl_np_1.obj"
    ]

    vertices_list, faces_list = load_mesh(mesh_path_list)

    intrinsics, extrinsics = load_cam("cam_path_TODO") # TODO

    image_points_list = project_points(vertices_list=vertices_list, intrinsics=intrinsics, extrinsics=extrinsics)

    for mesh_id, _ in enumerate(mesh_path_list):
        render(faces=faces_list[mesh_id], image_points=image_points_list[mesh_id], vertices=vertices_list[mesh_id], color=color_list[mesh_id])

    # show results
    cv2.imshow('result', image_canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()