import argparse
import util.util as util

def load_args():
    arg_parser = argparse.ArgumentParser()

    # Dataloader params
    arg_parser.add_argument('--input_dir', default='datasets/road2icy')
    arg_parser.add_argument('--num_channels', default=3, type=int)
    arg_parser.add_argument('--num_scales', default=3, type=int)
    arg_parser.add_argument('--scale_factor', default=2, type=int)
    arg_parser.add_argument('--base_resolution', default=64, type=int)
    arg_parser.add_argument('--num_threads', default=4, type=int)
    arg_parser.add_argument('--batch_size', default=1, type=int)
    arg_parser.add_argument('--serial_batches', action='store_true')
    arg_parser.add_argument('--preprocess', type=str, default='none')
    arg_parser.add_argument('--noise_mult', default=0.15, type=float)
    arg_parser.add_argument('--no_flip', action='store_true')
    arg_parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    arg_parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')

    # Scale params
    arg_parser.add_argument('--model_types', type=str, default='coarse,medium,fine')

    # CUT params
    arg_parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    arg_parser.add_argument('--name', type=str, default='experiment_name',
                        help='name of the experiment. It decides where to store samples and models')
    arg_parser.add_argument('--easy_label', type=str, default='experiment_name', help='Interpretable name')

    # Model params
    arg_parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')
    arg_parser.add_argument('--model', type=str, default='scale', help='chooses which model to use.')
    arg_parser.add_argument('--input_nc', type=int, default=3,
                        help='# of input image channels: 3 for RGB and 1 for grayscale')
    arg_parser.add_argument('--output_nc', type=int, default=3,
                        help='# of output image channels: 3 for RGB and 1 for grayscale')
    arg_parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    arg_parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
    arg_parser.add_argument('--netD', type=str, default='basic',
                        choices=['basic', 'n_layers', 'pixel', 'patch', 'tilestylegan2', 'stylegan2'],
                        help='specify discriminator architecture. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
    arg_parser.add_argument('--netG', type=str, default='resnet_9blocks',
                        choices=['resnet_9blocks', 'resnet_6blocks', 'unet_256', 'unet_128', 'stylegan2',
                                 'smallstylegan2', 'resnet_cat'], help='specify generator architecture')
    arg_parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
    arg_parser.add_argument('--normG', type=str, default='instance', choices=['instance', 'batch', 'none'],
                        help='instance normalization or batch normalization for G')
    arg_parser.add_argument('--normD', type=str, default='instance', choices=['instance', 'batch', 'none'],
                        help='instance normalization or batch normalization for D')
    arg_parser.add_argument('--init_type', type=str, default='xavier',
                        choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
    arg_parser.add_argument('--init_gain', type=float, default=0.02,
                        help='scaling factor for normal, xavier and orthogonal.')
    arg_parser.add_argument('--no_dropout', type=util.str2bool, nargs='?', const=True, default=True,
                        help='no dropout for the generator')
    arg_parser.add_argument('--no_antialias', action='store_true',
                        help='if specified, use stride=2 convs instead of antialiased-downsampling (sad)')
    arg_parser.add_argument('--no_antialias_up', action='store_true',
                        help='if specified, use [upconv(learned filter)] instead of [upconv(hard-coded [1,3,3,1] filter), conv]')
    arg_parser.add_argument('--lambda_GAN', type=float, default=1.0)
    arg_parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
    arg_parser.add_argument('--lambda_SSIM', type=float, default=1.0, help='weight for SSIM loss: SSIM(G(X), X)')
    arg_parser.add_argument('--lambda_RMSE', type=float, default=0.0, help='weight for RMSE loss: RMSE(G(X), X)')
    arg_parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False)
    arg_parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
    arg_parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                        type=util.str2bool, nargs='?', const=True, default=False,
                        help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
    arg_parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                        help='how to downsample the feature map')
    arg_parser.add_argument('--netF_nc', type=int, default=256)
    arg_parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
    arg_parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
    arg_parser.add_argument('--flip_equivariance',
                        type=util.str2bool, nargs='?', const=True, default=False,
                        help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
    arg_parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs with the initial learning rate')
    arg_parser.add_argument('--n_epochs_decay', type=int, default=200,
                        help='number of epochs to linearly decay learning rate to zero')
    arg_parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    arg_parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
    arg_parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    arg_parser.add_argument('--gan_mode', type=str, default='lsgan',
                        help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    arg_parser.add_argument('--pool_size', type=int, default=50,
                        help='the size of image buffer that stores previously generated images')
    arg_parser.add_argument('--lr_policy', type=str, default='linear',
                        help='learning rate policy. [linear | step | plateau | cosine]')
    arg_parser.add_argument('--lr_decay_iters', type=int, default=50,
                        help='multiply by a gamma every lr_decay_iters iterations')
    arg_parser.add_argument('--epoch_count', type=int, default=1,
                        help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    arg_parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    arg_parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
    arg_parser.set_defaults(pool_size=0)
    arg_parser.set_defaults(nce_idt=True, lambda_NCE=1.0)

    # Training params
    arg_parser.add_argument('--start_epoch', default=1, type=int)
    arg_parser.add_argument('--num_epochs', default=5, type=int)
    arg_parser.add_argument('--visuals_snapshots', default=10, type=int)
    arg_parser.add_argument('--print_freq', default=1, type=int)


    # Testing Params
    arg_parser.add_argument('--is_test', action='store_true')
    arg_parser.add_argument('--isTrain', type=util.str2bool, nargs='?', const=True, default=True)

    return arg_parser

