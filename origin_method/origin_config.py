# author:c19h
# datetime:2022/12/7 15:03
import os

class Config():
    curPath = os.path.abspath(__file__)
    upPath = os.path.split(curPath)[0]
    rootPath = os.path.split(upPath)[0]
    downsample_dir = os.path.join(rootPath, 'data/DVS128-Gait-Day/downsample_2000')
    origin_data_dir = os.path.join(rootPath, 'data/DVS128-Gait-Day/origin')
    data_dir = os.path.join(rootPath, 'data/DVS128-Gait-Day')
    graph_dir = os.path.join(data_dir, 'graph_origin_method')
    graph_train_dir = os.path.join(graph_dir, 'train')
    graph_test_dir = os.path.join(graph_dir, 'test')
    log_dir = os.path.join(rootPath, 'log')
    graph_train_log_path = os.path.join(log_dir, 'graph_train.log')
    cnn_train_log_path = os.path.join(log_dir, 'cnn_train_{}.log')
    model_dir = os.path.join(rootPath, 'trained_model')
    gcn_model_name = os.path.join(model_dir, 'EV_Gait_3DGraph_epoch_{}.pkl')
    batch_size = 16
    num_workers = 0
    split_graph_num = 1
    max_num = 1
    image_dir = os.path.join(data_dir, 'image')
    two_channels_counts_file = os.path.join(image_dir, 'two_channels_counts.hdf5')
    four_channels_file = os.path.join(image_dir, 'four_channels.hdf5')
    two_channels_time_file = os.path.join(image_dir, 'two_channels_time.hdf5')
    two_channels_counts_and_time_file = os.path.join(image_dir, 'two_channels_counts_and_time.hdf5')
    cnn_model_name = os.path.join(model_dir, 'EV_Gait_IMG_{}_epoch_{}.pkl')
