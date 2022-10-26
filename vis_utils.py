import open3d as o3d

def visualize_03d_pointcloud(o3d_point_cloud):
    o3d.visualization.draw_geometries([o3d_point_cloud])


def np_array_to_o3d_pointcloud(np_arr):
    o3d_pc= o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(np_arr)
    return o3d_pc


def visualize_np_pointcloud(np_arr):
    o3d_pc = np_array_to_o3d_pointcloud(np_arr)
    visualize_03d_pointcloud(o3d_pc)
