import os
import sys
import cv2
import argparse
import math

import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from logger import Logger as Log
import torchvision.transforms.functional as F

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import settings
from dataset import TrainValDataset, TestDataset
from model import Restormer
from cal_ssim import SSIM
from model import VGG

logger = settings.logger
os.environ['CUDA_VISIBLE_DEVICES'] = settings.device_id
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)
import numpy as np


def ssim_gray(self, imgA, imgB, gray_scale=True):
    if gray_scale:
        score, diff = ssim(cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY), cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY), full=True,
                           multichannel=True)
    # multichannel: If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each channel then averaged.
    else:
        score, diff = ssim(imgA, imgB, full=True, multichannel=True)
    return score


def psnr_gray(self, imgA, imgB):
    psnr_val = psnr(imgA, imgB)
    return psnr_val

def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def PSNR(img1, img2):
    b, _, _, _ = img1.shape
    img1 = np.clip(img1, 0, 255)
    img2 = np.clip(img2, 0, 255)
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse == 0:
        return 100

    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


class LSGANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(LSGANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                self.real_label_var = self.Tensor(input.size()).fill_(self.real_label)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                self.fake_label_var = self.Tensor(input.size()).fill_(self.fake_label)
            target_tensor = self.fake_label_var
        return target_tensor.cuda()

    def __call__(self, y_pred_fake, y_pred, target_is_real):
        y = self.get_target_tensor(y_pred_fake, True)
        y2 = self.get_target_tensor(y_pred_fake, False)
        if target_is_real:
            errD = (torch.mean((y_pred - torch.mean(y_pred_fake) - y) ** 2) + torch.mean(
                (y_pred_fake - torch.mean(y_pred) + y) ** 2)) / 2
            return errD
        else:
            errG = (torch.mean((y_pred - torch.mean(y_pred_fake) + y) ** 2) + torch.mean(
                (y_pred_fake - torch.mean(y_pred) - y) ** 2)) / 2
            return errG

class HingeGANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(HingeGANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                self.real_label_var = self.Tensor(input.size()).fill_(self.real_label)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                self.fake_label_var = self.Tensor(input.size()).fill_(self.fake_label)
            target_tensor = self.fake_label_var
        return target_tensor.cuda()

    def __call__(self, y_pred_fake, y_pred, target_is_real):
        target_tensor = self.get_target_tensor(y_pred_fake, target_is_real)
        if target_is_real:
            errD = (torch.mean(torch.nn.ReLU()(1.0 - (y_pred - torch.mean(y_pred_fake)))) + torch.mean(
                torch.nn.ReLU()(1.0 + (y_pred_fake - torch.mean(y_pred))))) / 2
            return errD
        else:
            errG = (torch.mean(torch.nn.ReLU()(1.0 + (y_pred - torch.mean(y_pred_fake)))) + torch.mean(
                torch.nn.ReLU()(1.0 - (y_pred_fake - torch.mean(y_pred))))) / 2
            return errG


class AverageStandardGAN(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(AverageStandardGAN, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.BCE_stable = torch.nn.BCEWithLogitsLoss().cuda()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                self.real_label_var = self.Tensor(input.size()).fill_(self.real_label)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                self.fake_label_var = self.Tensor(input.size()).fill_(self.fake_label)
            target_tensor = self.fake_label_var
        return target_tensor.cuda()

    def __call__(self, y_pred_fake, y_pred, target_is_real):
        target_tensor = self.get_target_tensor(y_pred_fake, target_is_real)
        y = self.get_target_tensor(y_pred_fake, True)
        y2 = self.get_target_tensor(y_pred_fake, False)
        if target_is_real:
            errD = ((self.BCE_stable(y_pred - torch.mean(y_pred_fake), y) + self.BCE_stable(
                y_pred_fake - torch.mean(y_pred), y2))) / 2
            return errD
        else:
            errG = ((self.BCE_stable(y_pred - torch.mean(y_pred_fake), y2) + self.BCE_stable(
                y_pred_fake - torch.mean(y_pred), y))) / 2
            return errG


class StandardGAN(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(StandardGAN, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.BCE_stable = torch.nn.BCEWithLogitsLoss().cuda()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                self.real_label_var = self.Tensor(input.size()).fill_(self.real_label)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                self.fake_label_var = self.Tensor(input.size()).fill_(self.fake_label)
            target_tensor = self.fake_label_var
        return target_tensor.cuda()

    def __call__(self, y_pred_fake, y_pred, target_is_real):
        target_tensor = self.get_target_tensor(y_pred_fake, target_is_real)
        if target_is_real:
            errD = self.BCE_stable(y_pred - y_pred_fake, self.get_target_tensor(y_pred, True))
            return errD
        else:
            errG = self.BCE_stable(y_pred_fake - y_pred, self.get_target_tensor(y_pred_fake, True))
            return errG


# class StandardGAN(nn.Module):
#     def __init__(self, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
#         super(StandardGAN, self).__init__()
#         self.real_label = target_real_label
#         self.fake_label = target_fake_label
#         self.real_label_var = None
#         self.fake_label_var = None
#         self.Tensor = tensor
#         self.BCE = torch.nn.BCELoss().cuda()
#
#     def get_target_tensor(self, input, target_is_real):
#         target_tensor = None
#         if target_is_real:
#             create_label = ((self.real_label_var is None) or
#                             (self.real_label_var.numel() != input.numel()))
#             if create_label:
#                 self.real_label_var = self.Tensor(input.size()).fill_(self.real_label)
#             target_tensor = self.real_label_var
#         else:
#             create_label = ((self.fake_label_var is None) or
#                             (self.fake_label_var.numel() != input.numel()))
#             if create_label:
#                 self.fake_label_var = self.Tensor(input.size()).fill_(self.fake_label)
#             target_tensor = self.fake_label_var
#         return target_tensor.cuda()
#
#     def __call__(self, y_pred_fake, y_pred, target_is_real):
#         target_tensor = self.get_target_tensor(y_pred_fake, target_is_real)
#         if target_is_real:
#             errD_real = self.BCE(y_pred, self.get_target_tensor(y_pred, True))
#
#             errD_fake = self.BCE(y_pred_fake, self.get_target_tensor(y_pred, False))
#
#             errD = errD_real + errD_fake
#
#             return errD
#         else:
#             errG = self.BCE(y_pred_fake, self.get_target_tensor(y_pred, True))
#             return errG

class Session:
    def __init__(self):
        self.log_dir = settings.log_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.log_dir)
        ensure_dir(settings.model_dir)
        ensure_dir('../log_test')
        # ensure_dir('../log_best')

        logger.info('set log dir as %s' % settings.log_dir)
        logger.info('set model dir as %s' % settings.model_dir)
        if len(settings.device_id) > 1:
            self.net = nn.DataParallel(Restormer()).cuda()
            # self.discriminator = nn.DataParallel(Discriminator()).cuda()
            # self.discriminator_local = nn.DataParallel(Discriminator_local()).cuda()

        else:
            self.net = Restormer().cuda()
            # self.discriminator = Discriminator().cuda()
            # self.discriminator_local = Discriminator_local().cuda()
        self.l1 = nn.L1Loss().cuda()
        self.l2 = nn.MSELoss().cuda()
        self.ssim_best = 0.0
        self.psnr_best = 0.0
        # if settings.GAN == 'HingeGAN':
        #     self.gan_loss = HingeGANLoss().cuda()
        # elif settings.GAN == 'LSGAN':
        #     self.gan_loss = LSGANLoss().cuda()
        # elif settings.GAN == 'StandardGAN':
        self.gan_loss = LSGANLoss().cuda()
        # elif settings.GAN == 'AverageStandardGAN':
        #     self.gan_loss = AverageStandardGAN().cuda()
        self.vgg = VGG().cuda()
        self.ssim = SSIM().cuda()
        self.log = Log()
        self.step = 0
        self.save_steps = settings.save_steps
        self.num_workers = settings.num_workers
        self.batch_size = settings.batch_size
        self.writers = {}
        self.dataloaders = {}
        self.opt_net = Adam(self.net.parameters(), lr=settings.lr)
        self.sche_net = MultiStepLR(self.opt_net, milestones=[settings.l1, settings.l2,settings.l3, settings.l4],
                                    gamma=0.5)

        # self.opt_discriminator = Adam(self.discriminator.parameters(), lr=settings.lr)
        # self.sche_discriminator = MultiStepLR(self.opt_discriminator, milestones=[settings.l1, settings.l2,settings.l3, settings.l4],
        #                                      gamma=0.5)
        #
        # self.opt_discriminator_local = Adam(self.discriminator_local.parameters(), lr=settings.lr)
        # self.sche_discriminator_local = MultiStepLR(self.opt_discriminator_local,
        #                                       milestones=[settings.l1, settings.l2, settings.l3, settings.l4],
        #                                       gamma=0.5)

        # self.discriminator_synimg = Discriminator_synimg().cuda()

    def tensorboard(self, name):
        self.writers[name] = SummaryWriter(os.path.join(self.log_dir, name + '.events'))
        return self.writers[name]

    def write(self, name, out):
        for k, v in out.items():
            self.writers[name].add_scalar(k, v, self.step)
        out['lr'] = self.opt_net.param_groups[0]['lr']
        out['step'] = self.step
        outputs = [
            "{}:{:.4g}".format(k, v)
            for k, v in out.items()
        ]
        logger.info(name + '--' + ' '.join(outputs))

    def get_dataloader(self, dataset_name):
        dataset = TrainValDataset(dataset_name, 0.5, 2)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                DataLoader(dataset, batch_size=self.batch_size,
                           shuffle=True, num_workers=self.num_workers, drop_last=True)
        return iter(self.dataloaders[dataset_name])

    def get_test_dataloader(self, dataset_name):
        dataset = TestDataset(dataset_name)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                DataLoader(dataset, batch_size=1,
                           shuffle=False, num_workers=1, drop_last=False)
        return self.dataloaders[dataset_name]

    def save_checkpoints_net(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        obj = {
            'net': self.net.state_dict(),
            'clock_net': self.step,
            'opt_net': self.opt_net.state_dict(),
            'ssim_best': self.ssim_best,
            'psnr_best': self.psnr_best,
        }
        torch.save(obj, ckp_path)

    def load_checkpoints_net(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            logger.info('Load checkpoint %s' % ckp_path)
            obj = torch.load(ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])
        self.opt_net.load_state_dict(obj['opt_net'])
        self.step = obj['clock_net']
        self.ssim_best = obj['ssim_best']
        self.psnr_best = obj['psnr_best']
        self.sche_net.last_epoch = self.step

    def save_checkpoints_discriminator(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        obj = {
            'net': self.discriminator.state_dict(),
            'clock_net': self.step,
            'opt_net': self.opt_discriminator.state_dict(),
        }
        torch.save(obj, ckp_path)

    def load_checkpoints_discriminator(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            logger.info('Load checkpoint %s' % ckp_path)
            obj = torch.load(ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.discriminator.load_state_dict(obj['net'])
        self.opt_discriminator.load_state_dict(obj['opt_net'])
        self.step = obj['clock_net']

        self.sche_discriminator.last_epoch = self.step

    def save_checkpoints_discriminator_local(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        obj = {
            'net': self.discriminator_local.state_dict(),
            'clock_net': self.step,
            'opt_net': self.opt_discriminator_local.state_dict(),
        }
        torch.save(obj, ckp_path)

    def load_checkpoints_discriminator_local(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            logger.info('Load checkpoint %s' % ckp_path)
            obj = torch.load(ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.discriminator_local.load_state_dict(obj['net'])
        self.opt_discriminator_local.load_state_dict(obj['opt_net'])
        self.step = obj['clock_net']

        self.sche_discriminator.last_epoch = self.step

    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))

    def pyramid_cl(self, clear, dehaze, haze):
        vgg_clear = self.vgg.forward(clear)
        vgg_dehaze = self.vgg.forward(dehaze)
        vgg_haze = self.vgg.forward(haze)
        loss_clear_dehaze_1 = self.l1(vgg_clear[0], vgg_dehaze[0])
        loss_dehaze_haze_1 = self.l1(vgg_haze[0], vgg_dehaze[0])
        loss_clear_dehaze_3 = self.l1(vgg_clear[1], vgg_dehaze[1])
        loss_dehaze_haze_3 = self.l1(vgg_haze[1], vgg_dehaze[1])
        loss_clear_dehaze_5 = self.l1(vgg_clear[2], vgg_dehaze[2])
        loss_dehaze_haze_5 = self.l1(vgg_haze[2], vgg_dehaze[2])
        loss_clear_dehaze_9 = self.l1(vgg_clear[3], vgg_dehaze[3])
        loss_dehaze_haze_9 = self.l1(vgg_haze[3], vgg_dehaze[3])
        loss_clear_dehaze_13 = self.l1(vgg_clear[4], vgg_dehaze[4])
        loss_dehaze_haze_13 = self.l1(vgg_haze[4], vgg_dehaze[4])
        loss1 = loss_clear_dehaze_1 / loss_dehaze_haze_1
        loss3 = loss_clear_dehaze_3 / loss_dehaze_haze_3
        loss5 = loss_clear_dehaze_5 / loss_dehaze_haze_5
        loss9 = loss_clear_dehaze_9 / loss_dehaze_haze_9
        loss13 = loss_clear_dehaze_13 / loss_dehaze_haze_13
        loss_total = 1 / 32 * loss1 + 1 / 16 * loss3 + 1 / 8 * loss5 + 1 / 4 * loss9 + 1 * loss13
        return loss_total

    def loss_vgg(self, restored, gt):
        vgg_gt = self.vgg.forward(gt)
        eval = self.vgg.forward(restored)
        loss_vgg = [self.l1(eval[m], vgg_gt[m]) for m in range(len(vgg_gt))]
        loss_vgg = sum(loss_vgg)
        return loss_vgg

    def inf_batch_net(self, name, batch):
        self.toogle_grad(self.net, True)
        # self.toogle_grad(self.discriminator, False)
        # self.toogle_grad(self.discriminator_local, False)
        # self.toogle_grad(self.dis_synreal, False)
        if name == 'train':
            self.net.zero_grad()
        if self.step == 0:
            self.print_network(self.net)

        # sample = {'O': O, 'B': B, 'O_gamma1': gamma_1, 'O_gamma2': gamma_2}

        O, B = batch['O'].cuda(), batch['B'].cuda()
        O, B = Variable(O, requires_grad=False), Variable(B, requires_grad=False)

        imgs = self.net(O)

        Syn_low_syn_enhanced = imgs
        loss =1  - self.ssim(Syn_low_syn_enhanced, B)

       # with torch.no_grad():
          #  mu = mu.mean(dim=0, keepdim=True)
           # momentum = 0.9
           # self.net.module.emau.mu *= momentum
           # self.net.module.emau.mu += mu * (1 - momentum)


        ssim1 = 1-loss
        psnr = PSNR(Syn_low_syn_enhanced.data.cpu().numpy() * 255, B.data.cpu().numpy() * 255)
        # self.log.add('lr','lr', self.opt_net.param_groups[0]['lr'], self.step)
        # self.log.add('loss', 'loss', loss, self.step)
        self.log.add('ssim', 'ssim', ssim1.data.cpu().numpy(), self.step)
        self.log.add('psnr', 'psnr', psnr, self.step)
        # print('loss',loss)
        # print('loss.size()', loss.size())
        if name == 'train':
            loss.backward()
            self.opt_net.step()
        # losses = {'L1loss2': loss}
        # ssimes = {'ssim1': ssim1}
        # losses.update(ssimes)
        # self.write(name, losses)

        losses = {'L1loss': loss}
        ssimes = {'ssim': ssim1}
        losses.update(ssimes)
        self.write(name, losses)

        return Syn_low_syn_enhanced



    def toogle_grad(self, model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)


    def save_image(self, name, img_lists):
        data, pred, label = img_lists
        pred = pred.cpu().data

        data, label, pred = data * 255, label * 255, pred * 255
        pred = np.clip(pred, 0, 255)
        h, w = pred.shape[-2:]
        gen_num = (1, 1)
        img = np.zeros((gen_num[0] * h, gen_num[1] * 3 * w, 3))
        for img_list in img_lists:
            for i in range(gen_num[0]):
                row = i * h
                for j in range(gen_num[1]):
                    idx = i * gen_num[1] + j
                    tmp_list = [data[idx], pred[idx], label[idx]]
                    for k in range(3):
                        col = (j * 3 + k) * w
                        tmp = np.transpose(tmp_list[k], (1, 2, 0))
                        img[row: row + h, col: col + w] = tmp
        img_file = os.path.join(self.log_dir, '%d_%s.jpg' % (self.step, name))
        cv2.imwrite(img_file, img)

    def ssim_gray(self, imgA, imgB, gray_scale=True):
        if gray_scale:
            score, diff = ssim(cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY), cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY), full=True, multichannel=True)
        # multichannel: If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each channel then averaged.
        else:
            score, diff = ssim(imgA, imgB, full=True, multichannel=True)
        return score

    def psnr_gray(self, imgA, imgB):
        psnr_val = psnr(cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY), cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY))
        return psnr_val


    def inf_batch_test(self, name, batch):
        O, B = batch['O'].cuda(), batch['B'].cuda()
        O, B = Variable(O, requires_grad=False), Variable(B, requires_grad=False)

        with torch.no_grad():
            imgs = self.net(O)
            out = imgs

        # l1_loss = self.l1(out, B)
        # ssim = self.ssim(out, B)
        # psnr = PSNR(out.data.cpu().numpy() * 255, B.data.cpu().numpy() * 255)

        gt = B[0].data.cpu().numpy() * 255
        gt = np.clip(gt, 0, 255)
        gt = np.transpose(gt, (1, 2, 0))

        img = out[0].data.cpu().numpy() * 255
        img = np.clip(img, 0, 255)
        img = np.transpose(img, (1, 2, 0))

        l1_loss = self.l1(out, B)
        ssim = self.ssim_gray(img.astype(np.uint8), gt.astype(np.uint8))
        psnr = self.psnr_gray(img.astype(np.uint8), gt.astype(np.uint8))

        # self.log.add('test_ssim', 'ssim', ssim, int(self.step/settings.one_epoch))
        # self.log.add('test_psnr', 'psnr', psnr, int(self.step/settings.one_epoch))
        # self.log.add('test_l1_loss', 'l1_loss', l1_loss, int(self.step / settings.one_epoch))
        # losses = {'L1 loss': l1_loss}
        # ssimes = {'ssim2': ssim}
        # losses.update(ssimes)

        return l1_loss.cpu().numpy(), ssim, psnr


def run_train(ckp_name_net='latest_net'):
    sess = Session()
    sess.load_checkpoints_net(ckp_name_net)
    # if settings.use_gan_synreal is True:
    #     sess.load_checkpoints_dissynreal(ckp_name_dissynreal)

    sess.tensorboard('train')
    dt_train = sess.get_dataloader('train')
    while sess.step < settings.total_step + 1:
        sess.sche_net.step()


        sess.net.train()

        try:
            batch_t = next(dt_train)
        except StopIteration:
            dt_train = sess.get_dataloader('train')
            batch_t = next(dt_train)
        pred = sess.inf_batch_net('train', batch_t)

        if sess.step % int(settings.one_epoch) == 0:
            sess.save_checkpoints_net('latest_net')


        if sess.step % settings.one_epoch == 0:
            sess.save_image('train', [batch_t['O'], pred, batch_t['B']])
            # sess.save_image('train-gamma', [batch_t['O'], batch_t['O_gamma1'], batch_t['O_gamma2']])
        # observe tendency of ssim, psnr and loss

        # ssim_all = 0
        # psnr_all = 0
        # loss_all = 0
        # num_all = 0
        # if sess.step % (settings.one_epoch) == 0:
        #     psnr_list = []
        #     ssim_list = []
        #     loss_list = []
        #     dt_val = sess.get_test_dataloader('test')
        #     sess.net.eval()
        #     for i, batch_v in enumerate(dt_val):
        #         loss, ssim, psnr = sess.inf_batch_test('test', batch_v)
        #         ssim_list.append(ssim)
        #         ssim_list.append(psnr)
        #         loss_list.append(loss)
        #         print(i)
        #         ssim_all = ssim_all + ssim
        #         psnr_all = psnr_all + psnr
        #         loss_all = loss_all + loss
        #         num_all = num_all + 1
        #     print('num_all:', num_all)
        #     loss_avg = loss_all / num_all
        #     ssim_avg = ssim_all / num_all
        #     psnr_avg = psnr_all / num_all
        #     var_psnr = np.mean(psnr_list)
        #     var_ssim = np.mean(ssim_list)
        #     var_loss = np.mean(loss_list)
        #     sess.log.add('test', 'ssim', ssim_avg, int(sess.step / settings.one_epoch))
        #     sess.log.add('test', 'psnr', psnr_avg, int(sess.step / settings.one_epoch))
        #     sess.log.add('test', 'loss', loss_avg, int(sess.step / settings.one_epoch))
        # 
        #     sess.log.add('test', 'var_ssim', var_ssim, int(sess.step / settings.one_epoch))
        #     sess.log.add('test', 'var_psnr', var_psnr, int(sess.step / settings.one_epoch))
        #     sess.log.add('test', 'var_loss', var_loss, int(sess.step / settings.one_epoch))
        #     logfile = open('../log_test/' + 'val' + '.txt', 'a+')
        #     epoch = int(sess.step / settings.one_epoch)
        #     logfile.write(
        #         'step  = ' + str(sess.step) + '\t'
        #                                       'epoch = ' + str(epoch) + '\t'
        #                                                                 'loss  = ' + str(loss_avg) + '\t'
        #                                                                                              'ssim  = ' + str(
        #             ssim_avg) + '\t'
        #                         'psnr  = ' + str(psnr_avg) + '\t'
        #                                                      '\n\n'
        #     )
        #     logfile.close()
        #     print('ssim_avg',ssim_avg)
        #     print('tea_ssim_best', sess.ssim_best)
        #     print('psnr_avg', psnr_avg)
        #     print('tea_psnr_best', sess.psnr_best)
        #     if ssim_avg >= sess.ssim_best and psnr_avg >= sess.psnr_best:
        #         logfile = open('../log_test/' + 'val_best' + '.txt', 'a+')
        #         epoch = int(sess.step / settings.one_epoch)
        #         logfile.write(
        #             'step  = ' + str(sess.step) + '\t'
        #                                           'epoch = ' + str(epoch) + '\t'
        #                                                                     'loss  = ' + str(loss_avg) + '\t'
        #                                                                                                  'ssim  = ' + str(
        #                 ssim_avg) + '\t'
        #                             'pnsr  = ' + str(psnr_avg) + '\t'
        #                                                          '\n\n'
        #         )
        #         sess.save_checkpoints_net('net_best_epoch')
        #         # if settings.use_gan is True:
        #         #     sess.save_checkpoints_discriminator('dis_best_epoch')
        #         # if settings.use_local_dis is True:
        #         #     sess.save_checkpoints_discriminator_local('dis_local_best_epoch')
        #         sess.ssim_best = ssim_avg
        #         sess.psnr_best = psnr_avg
        #         logfile.close()


        if sess.step % (500) == 0:
            sess.save_checkpoints_net('net_epoch')
            # if settings.use_gan_synreal is True:

            # if settings.use_gan_synreal is True:
            #     sess.save_checkpoints_dissynreal('dissynreal_%d_epoch' % int(sess.step / settings.one_epoch))
            logger.info('save model as net_%d_epoch' % int(sess.step / settings.one_epoch))
        sess.step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m1', '--model_1', default='latest_net')

    args = parser.parse_args(sys.argv[1:])
    run_train(args.model_1)

