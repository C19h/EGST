# author:c19h
# datetime:2022/12/7 10:39
import os

import numpy as np
import open3d as o3d
from tqdm import tqdm
import scipy.io as sio
from day_config import Config

origin_path_list = []
label_list = []
target_path_list = []
# iterate train and test
for train_test in os.listdir(Config.data_dir):
    # iterate each person
    for person in os.listdir(os.path.join(Config.data_dir, train_test)):
        # make corresponding person graph dir
        if not os.path.exists(os.path.join(Config.downsample_dir, train_test, person)):
            os.makedirs(os.path.join(Config.downsample_dir, train_test, person))
        for file in os.listdir(os.path.join(Config.data_dir, train_test, person)):
            origin_path_list.append(os.path.join(Config.data_dir, train_test, person, file))
            label_list.append(person)
            # faltten the graph files in one directory
            target_path_list.append(
                os.path.join(Config.downsample_dir, train_test, person, file.replace('txt', 'mat')))


def visulization(data):
    color = np.tile(data[:, 3], (3, 1)).T
    red_index = np.where(color[:, 0] == 1)
    color[red_index] = [255, 0, 0]
    blue_index = np.where(color[:, 0] == 0)
    color[blue_index] = [67, 142, 219]
    color /= 255
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([pcd])


def vector_angle(x, y):
    Lx = np.sqrt(x.dot(x))
    Ly = (np.sum(y ** 2, axis=1)) ** (0.5)
    cos_angle = np.sum(x * y, axis=1) / (Lx * Ly)
    angle = np.arccos(cos_angle)
    angle2 = angle * 360 / 2 / np.pi
    return angle2


def curvature_downsample(data):
    knn_num = 20  # 自定义参数值(邻域点数)
    angle_thre = 30  # 自定义参数值(角度值)
    N = 20  # 自定义参数值(每N个点采样一次)
    C = 30  # 自定义参数值(采样均匀性>N)

    point = data[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point)
    point_size = point.shape[0]
    tree = o3d.geometry.KDTreeFlann(pcd)
    o3d.geometry.PointCloud.estimate_normals(pcd, search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn_num))
    normal = np.asarray(pcd.normals)
    normal_angle = np.zeros((point_size))
    for i in range(point_size):
        [_, idx, dis] = tree.search_knn_vector_3d(point[i], knn_num + 1)
        current_normal = normal[i]
        knn_normal = normal[idx[1:]]
        normal_angle[i] = np.mean(vector_angle(current_normal, knn_normal))

    point_high = data[np.where(normal_angle >= angle_thre)]
    point_low = data[np.where(normal_angle < angle_thre)]
    pcd_high = o3d.geometry.PointCloud()
    pcd_high.points = o3d.utility.Vector3dVector(point_high[:, :3])
    pcd_high.colors = o3d.utility.Vector3dVector(point_high[:, [0, 3, 4]])
    pcd_low = o3d.geometry.PointCloud()
    pcd_low.points = o3d.utility.Vector3dVector(point_low[:, :3])
    pcd_low.colors = o3d.utility.Vector3dVector(point_low[:, [0, 3, 4]])
    pcd_high_down = o3d.geometry.PointCloud.uniform_down_sample(pcd_high, N)
    pcd_low_down = o3d.geometry.PointCloud.uniform_down_sample(pcd_low, C)
    pcd_finl = o3d.geometry.PointCloud()
    pcd_finl.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(pcd_high_down.points),
                                                                 np.asarray(pcd_low_down.points))))
    pcd_finl.colors = o3d.utility.Vector3dVector(np.concatenate((np.asarray(pcd_high_down.colors),
                                                                 np.asarray(pcd_low_down.colors))))
    cl, ind = pcd_finl.remove_statistical_outlier(nb_neighbors=40, std_ratio=0.1)
    inlier_cloud = pcd_finl.select_by_index(ind)
    downsample_data = np.asarray(inlier_cloud.points)
    downsample_feau = np.asarray(inlier_cloud.colors)
    total_events = np.concatenate([downsample_data, downsample_feau[:, [1, 4]].reshape((-1, 1))], axis=1)
    return total_events


def execute_downsample(origin_path, target_path, label, method='uniform_down_sample'):
    """

    :param origin_path:
    :param target_path:
    :param label:
    :param method: uniform_down_sample or curvature_down_sample
    """
    assert method in ['uniform_down_sample', 'voxel_down_sample',
                      'curvature_down_sample'], 'The {} not in down_sample methods'
    data = np.loadtxt(origin_path)
    tmin = data[:, 0].min()
    tmax = data[:, 0].max()
    data[:, 0] = 128 * (data[:, 0] - tmin) / (tmax - tmin)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(data[:, [0, 3, 3]])
    if method == 'uniform_down_sample':
        rd = pcd.uniform_down_sample(every_k_points=25)
        cl, ind = rd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.1, print_progress=False)
        inlier_cloud = rd.select_by_index(ind)
        # outlier_cloud = rd.select_by_index(ind, invert=True)
        downsample_data = np.asarray(inlier_cloud.points)
        downsample_feau = np.asarray(inlier_cloud.colors)
        total_events = np.concatenate([downsample_data, downsample_feau[:, 1].reshape((-1, 1))], axis=1)
        pass
    elif method == 'voxel_down_sample':
        rd = pcd.voxel_down_sample(voxel_size=6)
        cl, ind = rd.remove_statistical_outlier(nb_neighbors=40, std_ratio=0.01, print_progress=False)
        inlier_cloud = rd.select_by_index(ind)
        # outlier_cloud = rd.select_by_index(ind, invert=True)
        downsample_data = np.asarray(inlier_cloud.points)
        downsample_feau = np.asarray(inlier_cloud.colors)
        total_events = np.concatenate([downsample_data, downsample_feau[:, 1].reshape((-1, 1))], axis=1)
    else:
        total_events = curvature_downsample(data)
    # visulization(total_events)
    save_data = {"points": total_events}
    sio.savemat(target_path, save_data)
    pass


for i in tqdm(range(0, len(origin_path_list))):
    execute_downsample(origin_path_list[i], target_path_list[i], label_list[i], 'uniform_down_sample')