import open3d as o3d
import numpy as np
import math
import vis_utils
import glob

# source for ply models: https://people.sc.fsu.edu/~jburkardt/data/ply/ply.html

def create_pointcloud_from_mesh(mesh_path, nr_samples):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    point_cloud = mesh.sample_points_poisson_disk(number_of_points=nr_samples, init_factor=5)
    point_cloud.estimate_normals()
    normals = np.asarray(point_cloud.normals)
    points = np.asarray(point_cloud.points)
    bb = np.asarray(point_cloud.get_axis_aligned_bounding_box().get_box_points())
    bb_diag = 0
    for i in range(len(bb)):
        for j in range(len(bb)):
            pi = [bb[i][0], bb[i][1],bb[i][2]]
            pj = [bb[j][0], bb[j][1],bb[j][2]]
            diag = math.dist(pi, pj)
            if diag > bb_diag:
                bb_diag = diag
    return points, bb_diag, normals


def add_noise_to_cloud(point_cloud, deviation, name, normals, save_cloud=True, add_to_train_list=True):
    noise = np.random.normal(0, deviation, point_cloud.shape)
    noisy_pc = np.add(point_cloud, noise)
    if save_cloud:
        if deviation==0.0:
            file_name = "Dataset/Train/" + name + "_dev" + str(deviation) + "_normals"
            np.save(file_name, normals)
        file_name = "Dataset/Train/" + name + "_dev" + str(deviation)
        np.save(file_name, noisy_pc)
        if add_to_train_list:
            f = open("Dataset/Train/train.txt", "a+")
            f.write(file_name + "\n")
            f.close()
    return noisy_pc



if __name__ == '__main__':
    enable_vis = False
    #test = glob.glob("/home/decamargo/Documents/uni/miniproject/models/*")
    #print(test)

    file_path = "Dataset/Train/model_list.txt"
    file1 = open(file_path, 'r')
    lines = file1.readlines()
    for path in lines:
        path = path.replace("\n", "")
        name = path.split(".ply")[0].split("/")[-1]
        print("Processing: " + name)
        point_cloud, diag, normals = create_pointcloud_from_mesh(path, 100000)
        deviation = list(map(lambda i: i * diag, [0.0, 0.0025, 0.005, 0.01, 0.015, 0.025]))
        for d in deviation:
            noisy_point_cloud = add_noise_to_cloud(point_cloud, d, name, normals)
        if enable_vis:
            vis_utils.visualize_np_pointcloud(noisy_point_cloud)
            vis_utils.visualize_np_pointcloud(point_cloud)