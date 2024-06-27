import os
import cv2
import numpy as np
from numpy.random import RandomState
from torch.utils.data import Dataset

import settings 


def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


class TrainValDataset(Dataset):
    def __init__(self, name, gamma_1=1, gamma_2=1):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir_input = settings.data_dir_input
        self.root_dir_gt = settings.data_dir_gt
        self.mat_files = os.listdir(self.root_dir_input)
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2

        self.patch_size = settings.patch_size
        self.file_num = len(self.mat_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file_input = os.path.join(self.root_dir_input, file_name)

        img_file_gt = os.path.join(self.root_dir_gt, file_name)
        # img_pair = cv2.imread(img_file).astype(np.float32) / 255
        img_input = cv2.imread(img_file_input)
        img_gt = cv2.imread(img_file_gt)


        O, B = self.crop(img_input, img_gt, aug=False)



        O = O.astype(np.float32) / 255
        B = B.astype(np.float32) / 255



        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))



        sample = {'O': O, 'B': B}

        return sample

    def crop(self, img_input,img_gt, aug):
        patch_size = self.patch_size
        h, w, c = img_input.shape
        # w = int(ww / 2)

        # if aug:
        #     mini = - 1 / 4 * self.patch_size
        #     maxi =   1 / 4 * self.patch_size + 1
        #     p_h = patch_size + self.rand_state.randint(mini, maxi)
        #     p_w = patch_size + self.rand_state.randint(mini, maxi)
        # else:
        p_h, p_w = patch_size, patch_size

        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        O = img_input[r: r+p_h, c: c+p_w]
        B = img_gt[r: r+p_h, c: c+p_w]

        if aug:
            O = cv2.resize(O, (patch_size, patch_size))
            B = cv2.resize(B, (patch_size, patch_size))

        return O, B


    def crop_single(self, img, aug):
        patch_size = self.patch_size
        h, w, c = img.shape

        if aug:
            mini = - 1 / 4 * self.patch_size
            maxi =   1 / 4 * self.patch_size + 1
            p_h = patch_size + self.rand_state.randint(mini, maxi)
            p_w = patch_size + self.rand_state.randint(mini, maxi)
        else:
            p_h, p_w = patch_size, patch_size

        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        B = img[r: r+p_h, c: c+p_w]

        if aug:
            B = cv2.resize(B, (patch_size, patch_size))

        return B

    def flip(self, O, B):
        if self.rand_state.rand() > 0.5:
            O = np.flip(O, axis=1)
            B = np.flip(B, axis=1)
        return O, B

    def rotate(self, O, B):
        angle = self.rand_state.randint(-30, 30)
        patch_size = self.patch_size
        center = (int(patch_size / 2), int(patch_size / 2))
        M = cv2.getRotationMatrix2D(center, angle, 1)
        O = cv2.warpAffine(O, M, (patch_size, patch_size))
        B = cv2.warpAffine(B, M, (patch_size, patch_size))
        return O, B


class TestDataset(Dataset):
    def __init__(self, name, gamma_1=0.5, gamma_2=2):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = settings.data_dir_test
        print(self.root_dir)
        self.root_dir_gt = settings.data_dir_test_gt
        self.mat_files = os.listdir(self.root_dir)
        self.patch_size = settings.patch_size
        self.file_num = len(self.mat_files)
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        img_file_gt = os.path.join(self.root_dir_gt, file_name)
        print(img_file)
        img_pair = cv2.imread(img_file)
        img_pair_gt = cv2.imread(img_file_gt)
        print(img_pair.shape)
        h, ww, c = img_pair.shape
        w = int(ww / 2)
        h_8 = h % 1
        w_8 = w % 1
        O = img_pair[h_8:, w_8:]
        B = img_pair_gt[h_8:, w_8:]

        O = O.astype(np.float32) / 255
        B = B.astype(np.float32) / 255


        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))


        sample = {'O': O, 'B': B}

        return sample


class ShowDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir =settings.data_dir_test
        self.img_files = sorted(os.listdir(self.root_dir))
        self.file_num = len(self.img_files)
        # self.gamma_1 = settings.gamma1
        # self.gamma_2 = settings.gamma2
        # self.gamma_ori = settings.gamma_ori

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.img_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        print(img_file)
        img_file = os.path.join(self.root_dir, file_name)
        print(img_file)
        img_pair = cv2.imread(img_file)
        h, ww, c = img_pair.shape
        print(' h, ww, c = img_pair.shape', img_pair.shape)
        # if settings.pic_is_pair:
        #     w = int(ww / 2)
        #     h_8 = h % 1
        #     w_8 = w % 1
        #     O = img_pair[h_8:, w + w_8:2 * w]
        #     B = img_pair[h_8:, w_8:w]
        # else:
        w = ww
        h_8 = h % 1
        w_8 = w % 1
        O = img_pair
        B = img_pair

        # gamma_1 = gammaCorrection(O, self.gamma_1)
        # gamma_2 = gammaCorrection(O, self.gamma_2)
        # if self.gamma_ori != 1:
        #     O = gammaCorrection(O, self.gamma_ori)
        O = O.astype(np.float32) / 255
        B = B.astype(np.float32) / 255

        # gamma_1 = gamma_1.astype(np.float32) / 255
        # gamma_2 = gamma_2.astype(np.float32) / 255

        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))

        # gamma_1 = np.transpose(gamma_1, (2, 0, 1))
        # gamma_2 = np.transpose(gamma_2, (2, 0, 1))

        sample = {'O': O, 'B': B,'file_name':file_name[:-4]}
        return sample


if __name__ == '__main__':
    dt = TrainValDataset('val')
    print('TrainValDataset')
    for i in range(10):
        smp = dt[i]
        for k, v in smp.items():
            print(k, v.shape, v.dtype, v.mean())

    print()
    dt = TestDataset('test')
    print('TestDataset')
    for i in range(10):
        smp = dt[i]
        for k, v in smp.items():
            print(k, v.shape, v.dtype, v.mean())

    print()
    print('ShowDataset')
    dt = ShowDataset('test')
    for i in range(10):
        smp = dt[i]
        for k, v in smp.items():
            print(k, v.shape, v.dtype, v.mean())
