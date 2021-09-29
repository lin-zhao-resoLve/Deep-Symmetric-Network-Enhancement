import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss
from models.modules.Quantization import Quantization

logger = logging.getLogger('base')

class IRNModel(BaseModel):
    def __init__(self, opt):
        super(IRNModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt

        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        self.Quantization = Quantization()

        if self.is_train:
            self.netG.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])


            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.ref_L = data['LQ'].to(self.device)  # LQ
        self.real_H = data['GT'].to(self.device)  # GT

    def gaussian_batch(self, dims):
        return torch.randn(tuple(dims)).to(self.device)

    def loss_forward(self, out, y):
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)

        # z = z.reshape([out.shape[0], -1])
        # l_forw_ce = self.train_opt['lambda_ce_forw'] * torch.sum(z**2) / z.shape[0]

        return l_forw_fit

    def loss_backward_HR(self, x, y):
        x_samples = self.netG(x=y, rev=True)
        # x_samples_image = x_samples[:, :3, :, :]
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(x, x_samples)

        return l_back_rec

    def loss_backward_LR(self, x, y):
        x_samples = self.netG(x=y, rev=False)
        #x_samples_image = x_samples[:, :4, :, :]
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(x, x_samples)

        return l_back_rec


    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()

        # forward downscaling
        self.input_f = self.real_H
        # self.output_f = self.netG(x=self.input_f)
        self.input_b = self.ref_L
        self.output_b = self.netG(x=self.input_b, rev=True)
        HR_real = self.real_H.detach()

        l_forw_fit_HR = self.loss_forward(self.output_b, HR_real)


        # backward upscaling
        #y_ = torch.cat((LR, LR_r), dim=1)
        # HR = self.Quantization(self.output_b[:, :3, :, :])
        HR = self.Quantization(self.real_H[:, :3, :, :])
        l_back_rec_LR = self.loss_backward_LR(self.input_b, HR)

        # total loss
        loss = l_back_rec_LR + l_forw_fit_HR
        loss.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer_G.step()

        # set log
        self.log_dict['l_back_rec_LR'] = l_back_rec_LR.item()
        self.log_dict['l_forw_fit_HR'] = l_forw_fit_HR.item()

    def _overlap_crop_forward_f(self, x, shave=10, min_size=4000000):
        """
        chop for less memory consumption during test
        """
        n_GPUs = 1
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)

                sr_batch_temp = self.netG(x=lr_batch)[:, :3, :, :]

                # if isinstance(sr_batch_temp, list):
                #     sr_batch = sr_batch_temp[-1]
                # else:
                #     sr_batch = sr_batch_temp

                sr_list.extend(sr_batch_temp.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self._overlap_crop_forward_f(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
                ]

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def _overlap_crop_forward_rev(self, x, shave=10, min_size=4000000):
        """
        chop for less memory consumption during test
        """
        n_GPUs = 1
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)

                sr_batch_temp = self.netG(x=lr_batch, rev=True)[:, :3, :, :]

                # if isinstance(sr_batch_temp, list):
                #     sr_batch = sr_batch_temp[-1]
                # else:
                #     sr_batch = sr_batch_temp

                sr_list.extend(sr_batch_temp.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self._overlap_crop_forward_rev(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output
    def test(self):
        self.input = self.real_H

        self.netG.eval()
        with torch.no_grad():
            # forward_func_f = self._overlap_crop_forward_f
            # forward_func_rev = self._overlap_crop_forward_rev
            # self.forw_L = forward_func_f(x=self.input)[:, :3, :, :]
            # self.forw_L = self.Quantization(self.forw_L)
            # self.fake_H = forward_func_rev(x=self.ref_L)[:, :3, :, :]
            # self.forw_L = self.netG(x=self.input)[:, :3, :, :]
            # self.forw_L = self.Quantization(self.forw_L)
            self.fake_H = self.netG(x=self.ref_L, rev=True)[:, :3, :, :]

        self.netG.train()

    def downscale(self, HR_img):
        self.netG.eval()
        with torch.no_grad():
            LR_img = self.netG(x=HR_img)[:, :3, :, :]
            LR_img = self.Quantization(self.forw_L)
        self.netG.train()

        return LR_img

    def upscale(self, LR_img, scale, gaussian_scale=1):
        Lshape = LR_img.shape
        zshape = [Lshape[0], Lshape[1] * (scale**2 - 1), Lshape[2], Lshape[3]]
        y_ = torch.cat((LR_img, gaussian_scale * self.gaussian_batch(zshape)), dim=1)

        self.netG.eval()
        with torch.no_grad():
            HR_img = self.netG(x=y_, rev=True)[:, :3, :, :]
        self.netG.train()

        return HR_img

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LR_ref'] = self.ref_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        # out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
