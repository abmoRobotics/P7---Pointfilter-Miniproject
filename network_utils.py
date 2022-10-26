import numpy as np
import torch

"""Contains helper functions regarding the NN"""

def adjust_learning_rate(optimizer, epoch, opt):
    lr_scheduler(optimizer, epoch, opt.lr)


def lr_scheduler(optimizer, epoch, init_lr):
    if epoch > 36:
        init_lr *= 0.5e-3
    elif epoch > 32:
        init_lr *= 1e-3
    elif epoch > 24:
        init_lr *= 1e-2
    elif epoch > 16:
        init_lr *= 1e-1
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr


def compute_bilateral_loss_with_repulsion(pred_point, gt_patch_pts, gt_patch_normals, support_radius, support_angle, alpha):
    # Our Loss
    print("Predicted points shape: " + str(pred_point.shape))
    pred_point = pred_point.unsqueeze(1).repeat(1, gt_patch_pts.size(1), 1)
    print("Predicted points shape after unsqueeze: " + str(pred_point.shape))
    print("Gt points shape: " + str(gt_patch_pts.shape))
    print("Support radius: " + str(support_radius.shape))
    print("Support angle: " + str(support_angle))
    print("Repulse alpha: " + str(alpha))
    dist_square = ((pred_point - gt_patch_pts) ** 2).sum(2)
    weight_theta = torch.exp(-1 * dist_square / (support_radius ** 2))
    nearest_idx = torch.argmin(dist_square, dim=1)
    pred_point_normal = torch.cat([gt_patch_normals[i, index, :] for i, index in enumerate(nearest_idx)])
    pred_point_normal = pred_point_normal.view(-1, 3)
    pred_point_normal = pred_point_normal.unsqueeze(1)
    pred_point_normal = pred_point_normal.repeat(1, gt_patch_normals.size(1), 1)
    normal_proj_dist = (pred_point_normal * gt_patch_normals).sum(2)
    weight_phi = torch.exp(-1 * ((1 - normal_proj_dist) / (1 - np.cos(support_angle)))**2)
    # # avoid divided by zero
    weight = weight_theta * weight_phi + 1e-12
    weight = weight / weight.sum(1, keepdim=True)
    # key loss
    project_dist = torch.abs(((pred_point - gt_patch_pts) * gt_patch_normals).sum(2))
    imls_dist = (project_dist * weight).sum(1)
    # repulsion loss
    max_dist = torch.max(dist_square, 1)[0]
    # final loss
    dist = torch.mean((alpha * imls_dist) + (1 - alpha) * max_dist)

    return dist