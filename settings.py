
import os
import logging


#######################  网络参数设置　　############################################################
channel = 32 # 16, 20, 24, 28, 32, 36, 48, 52, 56
loss = 'ssim'  # l1, l2, ssim
########################################################################################
aug_data = False # Set as False for fair comparison





patch_size = 128
pic_is_pair = True  # input picture is pair or single

lr = 0.0001


data_dir_input = './dataset/train_data/train/input'
data_dir_gt = './dataset/train_data/train/target'

data_dir_test = './dataset/Rain200L/test/input'
#data_dir_test_gt = './dataset/Rain200L/test/target'


if pic_is_pair is False:
    data_dir = '/data2/wangcong/dataset/haze_dataset'
log_dir = '../logdir'
imgs_dir = '../logdir/imgs'
show_dir = '../showdir'
model_dir = '../train_models'
show_dir_feature = '../showdir_feature'

log_level = 'info'
model_path = os.path.join(model_dir, 'latest_net')
save_steps = 200

num_workers = 12

num_GPU = 1

device_id = '0'
# device_id = None


epoch = 200
batch_size = 2

if pic_is_pair:
    root_dir = data_dir_input
    mat_files = os.listdir(root_dir)
    num_datasets = len(mat_files)
    l1 = int(1/5 * epoch * num_datasets / batch_size)
    l2 = int(2/5 * epoch * num_datasets / batch_size)
    l3 = int(3/5 * epoch * num_datasets / batch_size)
    l4 = int(4/5 * epoch * num_datasets / batch_size)
    one_epoch = int(num_datasets/batch_size)
    total_step = int((epoch * num_datasets)/batch_size)

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


