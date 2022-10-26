import vis_utils
import numpy as np

test_path = "/home/decamargo/Documents/uni/miniproject/Pointfilter-Miniproject/Dataset/Test/"
res_path = "/home/decamargo/Documents/uni/miniproject/Pointfilter-Miniproject/Dataset/Results/"

pc_noisy_path = test_path + "sphere_dev1.0938961202360427.npy"
pc_filtered_path = res_path + "sphere_dev1.0938961202360427_pred_iter_1.npy"

pc_noisy = np.load(pc_noisy_path)
pc_filtered = np.load(pc_filtered_path)
print(pc_noisy)
print(pc_filtered)
vis_utils.visualize_np_pointcloud(pc_noisy)
vis_utils.visualize_np_pointcloud(pc_filtered)