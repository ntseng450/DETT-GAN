import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import pytorch_ssim
from torch.autograd import Variable
import torch.nn as nn


class ScaleModel(BaseModel):
    """Generator and discriminator pair for a single scale for the multi-scale architecture proposed
    in the paper. The architecture uses a CUT-GAN, with different discriminator/generator parameters and
    receptive field sizes per scale and content preservation loss for scale < N-1.

    The code borrows heavily from the PyTorch implementation of CycleGAN and CUT-GAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    https://github.com/taesungp/contrastive-unpaired-translation
    """
    def __init__(self, opt, scale):
        BaseModel.__init__(self, opt)

        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'SSIM', 'RMSE']
        self.visual_names = ['input_A', 'fake_B', 'real_B', 'original_A']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:
            self.model_names = ['G']

        # define network for the current scale
        model_specs = opt.model_types.split(',')
        scale_idx = scale - 1
        if model_specs[scale_idx] == "coarse":
            self.netG = networks.define_G(opt.input_nc, opt.output_nc, 32, 'resnet_4blocks', opt.normG, not opt.no_dropout, 
            opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        elif model_specs[scale_idx] == "medium":
            self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'resnet_6blocks', opt.normG, not opt.no_dropout, 
            opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        else:
            self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            if model_specs[scale_idx] == "coarse" or model_specs[scale_idx] == "medium":
                self.netD = networks.define_D(opt.output_nc, 32, model_specs[scale_idx], opt.n_layers_D, opt.normD, 
                opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            else:
                self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []
            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.criterionSSIM = pytorch_ssim.SSIM(window_size=11)
            self.criterionRMSE = nn.MSELoss()

            # define optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """ Data dependent initialize from CUT-GAN training methodology, weights are initialized from the
        shape of netG's encoder.
        """
        self.set_input(data)
        bs_per_gpu = self.input_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.input_A = self.input_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()
        if self.opt.isTrain:
            self.compute_D_loss().backward()
            self.compute_G_loss().backward()
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Get upsampled output from previous scale with the corresponding original image it came from, and
        get target image.
        """
        self.input_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.original_A = input['orig'].to(self.device)
        self.image_paths = input['A_paths']


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.nce_idt and self.opt.isTrain:
            if self.opt.condition_original:
                self.real = torch.cat((self.input_A, self.real_B, self.original_A), dim=0)
            else:
                self.real = torch.cat((self.input_A, self.real_B), dim=0)
        else:
            self.real = self.input_A
        # self.real = torch.cat((self.input_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.input_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.input_A.size(0)]
        if self.opt.nce_idt and not self.opt.condition_original:
            self.idt_B = self.fake[self.input_A.size(0):]
        if self.opt.nce_idt and self.opt.condition_original:
            self.idt_B = self.fake[self.input_A.size(0):2]
            self.idt_A = self.fake[2:]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN, NCE, and content preservation loss for the generator"""
        fake = self.fake_B
        # Calculate generator GAN loss
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        # Calculate NCE loss
        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.input_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y)
            if self.opt.condition_original:
                self.loss_NCE_original = self.calculate_NCE_loss(self.original_A, self.idt_A)
                loss_NCE_total = (loss_NCE_both + self.loss_NCE_original) / 3
            else:
                loss_NCE_total = loss_NCE_both * 0.5
        else:
            loss_NCE_total = self.loss_NCE

        # Calculate content preservation loss
        if self.opt.lambda_SSIM > 0.0:
            self.loss_SSIM = self.calculate_SSIM_loss(self.original_A, self.fake_B)
        else:
            self.loss_SSIM = 0.0

        if self.opt.lambda_RMSE > 0.0:
            self.loss_RMSE = self.calculate_RMSE_loss(self.original_A, self.fake_B)
        else:
            self.loss_RMSE = 0.0

        # Calculate objective loss function
        self.loss_G = self.loss_G_GAN + loss_NCE_total + self.loss_SSIM + self.loss_RMSE
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def calculate_SSIM_loss(self, original_src, generated):
        src = Variable(original_src, requires_grad=False)
        fake = Variable(generated, requires_grad=True)
        
        ssim = 1 - self.criterionSSIM(src, fake)
        ssim = ssim * self.opt.lambda_SSIM
        return ssim 

    def calculate_RMSE_loss(self, original_src, generated):
        rmse = torch.sqrt(self.criterionRMSE(original_src, generated))
        rmse = rmse * self.opt.lambda_RMSE
        return rmse