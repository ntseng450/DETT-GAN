https://distill.pub/2019/computing-receptive-fields/

HERE
torch.Size([1, 64, 63, 63])
torch.Size([1, 64, 32, 32])
torch.Size([1, 128, 31, 31])
torch.Size([1, 128, 16, 16])
torch.Size([1, 256, 15, 15])
torch.Size([1, 256, 8, 8])
torch.Size([1, 512, 7, 7])
torch.Size([1, 1, 6, 6])
HERE

torch.Size([1, 64, 127, 127])
torch.Size([1, 64, 64, 64])
torch.Size([1, 128, 63, 63])
torch.Size([1, 128, 32, 32])
torch.Size([1, 256, 31, 31])
torch.Size([1, 256, 16, 16])
torch.Size([1, 512, 15, 15])
torch.Size([1, 1, 14, 14])


torch.Size([1, 64, 255, 255])
torch.Size([1, 64, 128, 128])
torch.Size([1, 128, 127, 127])
torch.Size([1, 128, 64, 64])
torch.Size([1, 256, 63, 63])
torch.Size([1, 256, 32, 32])
torch.Size([1, 512, 31, 31])
torch.Size([1, 1, 30, 30])
# base_res=64
# scale_factor=2

# 3x64x64
# conv2d: ndf=16, s=1, k=5, p=1
# 16x62x62
# downsample: ndf=16, s=3, k=4, p=1
# 16x31x31

# l=1 (4-1)(1)
# l=2 (3-1)(2*1)
# l=3 (4-1)(1*2*1)
# l=4 (3-1)(2*1*2*1)
# l=5 (4-1)(4)
# l=6 (3-1)(8)
# l=7 (4-1)(8)
# l=8 (4-1)(8)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # output = ((dim + 2*padding - kernel) / (stride)) + 1

        kw = 4
        padw = 1
        if(no_antialias):
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            # 3x64x64
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True), Downsample(ndf)]
            # after conv2d: torch.Size([1, 64, 63, 63])
            # after downsample: torch.Size([1, 64, 32, 32])
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if(no_antialias):
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                # after conv2d 1: torch.Size([1, 128, 31, 31])
                # after Downsample 1: torch.Size([1, 128, 16, 16])
                # after conv2d 2: torch.Size([1, 256, 15, 15])
                # after Downsample 2: torch.Size([1, 256, 8, 8])
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        # after conv2d: torch.Size([1, 512, 7, 7])

        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

         # after conv2d torch.Size([1, 1, 6, 6])
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)



class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        # pad_sizes = [1, 1, 1, 1]
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        # off = 0
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

# ----------------- Options ---------------
#                  CUT_mode: CUT                           
#                batch_size: 1                             
#                     beta1: 0.5                           
#                     beta2: 0.999                         
#           checkpoints_dir: ./checkpoints                 
#            continue_train: False                         
#                 crop_size: 256                           
#                  dataroot: ./datasets/cmvid2icy          	[default: placeholder]
#              dataset_mode: unaligned                     
#                 direction: AtoB                          
#               display_env: main                          
#              display_freq: 400                           
#                display_id: 0                             	[default: None]
#             display_ncols: 4                             
#              display_port: 8097                          
#            display_server: http://localhost              
#           display_winsize: 256                           
#                easy_label: experiment_name               
#                     epoch: latest                        
#               epoch_count: 1                             
#           evaluation_freq: 5000                          
#         flip_equivariance: False                         
#                  gan_mode: lsgan                         
#                   gpu_ids: 0                             
#                 init_gain: 0.02                          
#                 init_type: xavier                        
#                  input_nc: 3                             
#                   isTrain: True                          	[default: None]
#                lambda_GAN: 1.0                           
#                lambda_NCE: 1.0                           
#                 load_size: 286                           
#                        lr: 0.0002                        
#            lr_decay_iters: 50                            
#                 lr_policy: linear                        
#          max_dataset_size: inf                           
#                     model: cut                           
#                  n_epochs: 200                           
#            n_epochs_decay: 200                           
#                n_layers_D: 3                             
#                      name: cmvid2icy_CUT                 	[default: experiment_name]
#                     nce_T: 0.07                          
#                   nce_idt: True                          
# nce_includes_all_negatives_from_minibatch: False                         
#                nce_layers: 0,4,8,12,16                   
#                       ndf: 64                            
#                      netD: basic                         
#                      netF: mlp_sample                    
#                   netF_nc: 256                           
#                      netG: resnet_9blocks                
#                       ngf: 64                            
#              no_antialias: False                         
#           no_antialias_up: False                         
#                no_dropout: True                          
#                   no_flip: False                         
#                   no_html: False                         
#                     normD: instance                      
#                     normG: instance                      
#               num_patches: 256                           
#               num_threads: 4                             
#                 output_nc: 3                             
#                     phase: train                         
#                 pool_size: 0                             
#                preprocess: resize_and_crop               
#           pretrained_name: None                          
#                print_freq: 100                           
#          random_scale_max: 3.0                           
#              save_by_iter: False                         
#           save_epoch_freq: 5                             
#          save_latest_freq: 5000                          
#            serial_batches: False                         
# stylegan2_G_num_downsampling: 1                             
#                    suffix:                               
#          update_html_freq: 1000                          
#                   verbose: False                         
# ----------------- End -------------------

