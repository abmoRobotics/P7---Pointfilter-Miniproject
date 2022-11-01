import utils.vis_utils as vis_utils
import numpy as np

test_path = "./Dataset/Test/"
train_path = "./Dataset/Train/"
res_path = "./Dataset/Results/"

pc_noisy_path = test_path + "A110.PLY_dev7.85381677924566.npy"
pc_filtered_path = res_path + "A110.PLY_dev7.85381677924566_pred_iter_0.npy"

pc_input = train_path + ".npy"
pc_input_noisy = train_path + ".npy"

pc_noisy = np.load(pc_noisy_path)
pc_filtered = np.load(pc_filtered_path)
print(pc_noisy)
print(pc_filtered)
vis_utils.visualize_np_pointcloud(pc_noisy)
vis_utils.visualize_np_pointcloud(pc_filtered)