import numpy as np
import os
import argparse
import random
import torch

import configs
import util

from ipdb import set_trace as debug


def set():
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',              type=str, default='train', help='Train, test phase')
    parser.add_argument("--experiments-path",   type=str, default='experiments')
    parser.add_argument("--problem-name",   type=str)
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--gpu",            type=int,   default=0,        help="GPU device")
    parser.add_argument("--load",           type=str,   default=None,     help="load the checkpoints")
    parser.add_argument("--dir",            type=str,   default=None,     help="directory name to save the experiments under results/")
    parser.add_argument("--group",          type=str,   default='0',      help="father node of directionary for saving checkpoint")
    parser.add_argument("--name",           type=str,   default='debug',  help="son node of directionary for saving checkpoint")
    parser.add_argument("--log-fn",         type=str,   default=None,     help="name of tensorboard logging")
    parser.add_argument("--log-tb",         action="store_true",          help="logging with tensorboard")
    parser.add_argument("--cpu",            action="store_true",          help="use cpu device")

    # --------------- SB model ---------------
    parser.add_argument("--t0",             type=float, default=1e-2,     help="time integral start time")
    parser.add_argument("--T",              type=float, default=1.,       help="time integral end time")
    parser.add_argument("--interval",       type=int,   default=100,      help="number of interval")
    parser.add_argument("--forward-net",    type=str,   choices=['toy','Unet','ncsnpp'], help="model class of forward nonlinear drift")
    parser.add_argument("--backward-net",   type=str,   choices=['toy','Unet','ncsnpp'], help="model class of backward nonlinear drift")
    parser.add_argument("--sde-type",       type=str,   default='ve', choices=['ve', 'vp', 'simple'])
    parser.add_argument("--sigma-max",      type=float, default=50,       help="max diffusion for VESDE")
    parser.add_argument("--sigma-min",      type=float, default=0.01,     help="min diffusion for VESDE")
    parser.add_argument("--beta-max",       type=float, default=20,       help="max diffusion for VPSDE")
    parser.add_argument("--beta-min",       type=float, default=0.1,      help="min diffusion for VPSDE")
    parser.add_argument("--var",            type=float, default=1.,       help='diffusion coefficient in simple SDEs')

    #---------------- Divide n Conquer settings ----------
    parser.add_argument('--log-SNR-max', type=float, default=10, help='SNR value at time t0.')
    parser.add_argument('--log-SNR-min', type=float, default=-10, help='SNR value at time 1.')
    parser.add_argument('--max-num-intervals', type=int, default=2**3, help='num intervals')
    parser.add_argument('--num-outer-iterations', type=int, default=4, help='outer loop iterations.')
    parser.add_argument('--num-inner-iterations', type=int, default=150, help='outer loop iterations.')
    parser.add_argument('--inner_it_save_freq', type=int, default=10)
    parser.add_argument('--policy-updates', type=int, default=25, help='alternating policy updates')
    parser.add_argument('--base-discretisation', type=int, default=8, help='base discretisation')
    
    # --------------- SB training & sampling (corrector) ---------------
    parser.add_argument("--training-scheme", type=str, default='standard', help='training schem. Options=[standard, divideNconquer]')
    parser.add_argument("--train-method",   type=str, default=None,       help="algorithm for training SB" )
    parser.add_argument("--use-arange-t",   action="store_true",          help="[sb alternate train] use full timesteps for training")
    parser.add_argument("--reuse-traj",     action="store_true",          help="[sb alternate train] reuse the trajectory from sampling")
    parser.add_argument("--use-corrector",  action="store_true",          help="[sb alternate train] enable corrector during sampling")
    parser.add_argument("--train-bs-x",     type=int,                     help="[sb alternate train] batch size for sampling data")
    parser.add_argument("--train-bs-t",     type=int,                     help="[sb alternate train] batch size for sampling timestep")
    parser.add_argument("--num-stage",      type=int,                     help="[sb alternate train] number of stage")
    parser.add_argument("--num-epoch",      type=int,                     help="[sb alternate train] number of training epoch in each stage")
    parser.add_argument("--num-corrector",  type=int, default=1,          help="[sb alternate train] number of corrector steps")
    parser.add_argument("--snr",            type=float,                   help="[sb alternate train] signal-to-noise ratio")
    parser.add_argument("--eval-itr",       type=int, default=200,        help="[sb joint train] frequency of evaluation")
    parser.add_argument("--samp-bs",        type=int,                     help="[sb train] batch size for all trajectory sampling purposes")
    parser.add_argument("--num-itr",        type=int,                     help="[sb train] number of training iterations (for each epoch)")

    parser.add_argument("--DSM-warmup",     action="store_true",          help="[dsm warmup train] enable dsm warmup at 1st stage")
    parser.add_argument("--train-bs-x-dsm", type=int,                     help="[dsm warmup train] batch size for sampling data")
    parser.add_argument("--train-bs-t-dsm", type=int,                     help="[dsm warmup train] batch size for sampling timestep")
    parser.add_argument("--num-itr-dsm",    type=int,                     help="[dsm warmup train] number of training iterations for DSM warmup")

    # --------------- optimizer and loss ---------------
    parser.add_argument("--lr",             type=float,                   help="learning rate")
    parser.add_argument("--lr-f",           type=float, default=None,     help="learning rate for forward network")
    parser.add_argument("--lr-b",           type=float, default=None,     help="learning rate for backward network")
    parser.add_argument("--lr-gamma",       type=float, default=1.0,      help="learning rate decay ratio")
    parser.add_argument("--lr-step",        type=int,   default=1000,     help="learning rate decay step size")
    parser.add_argument("--l2-norm",        type=float, default=0.0,      help="weight decay rate")
    parser.add_argument("--optimizer",      type=str,   default='AdamW',  help="optmizer")
    parser.add_argument("--grad-clip",      type=float, default=None,     help="clip the gradient")
    parser.add_argument("--noise-type",     type=str,   default='gaussian', choices=['gaussian','rademacher'], help='choose noise type to approximate Trace term')

    # ---------------- evaluation ----------------
    parser.add_argument("--FID-freq",       type=int,   default=0,        help="FID frequency w.r.t stages")
    parser.add_argument("--snapshot-freq",  type=int,   default=0,        help="snapshot frequency w.r.t stages")
    parser.add_argument("--ckpt-freq",      type=int,   default=0,        help="checkpoint saving frequency w.r.t stages")
    parser.add_argument("--FID-ckpt",       type=str,   default=None,     help="manually set ckpt path")
    parser.add_argument("--num-FID-sample", type=int,   default=10000,    help="number of sample for computing FID")
    parser.add_argument("--compute-FID",    action="store_true",          help="flag: evaluate FID")
    parser.add_argument("--compute-NLL",    action="store_true",          help="flag: evaluate NLL")

    problem_name = parser.parse_args().problem_name
    default_config, model_configs = {
        'gmm':          configs.get_gmm_default_configs,
        'checkerboard': configs.get_checkerboard_default_configs,
        'moon-to-spiral':configs.get_moon_to_spiral_default_configs,
        'cifar10':      configs.get_cifar10_default_configs,
        'celebA64':     configs.get_celebA64_default_configs,
        'celebA32':     configs.get_celebA32_default_configs,
        'mnist':        configs.get_mnist_default_configs,
    }.get(problem_name)()
    parser.set_defaults(**default_config)

    opt = parser.parse_args()

    # ========= seed & torch setup =========
    if opt.seed is not None:
        # https://github.com/pytorch/pytorch/issues/7068
        seed = opt.seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True

    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_tensor_type('torch.FloatTensor')
    # torch.autograd.set_detect_anomaly(True)
    
    # ========= auto setup & path handle =========
    opt.device='cuda:'+str(opt.gpu)
    #opt.device='cpu'
    opt.model_configs = model_configs
    if opt.lr is not None:
        opt.lr_f, opt.lr_b = opt.lr, opt.lr

    if opt.compute_NLL or opt.compute_FID:
        opt.DSM_warmup = False
        opt.train_method = None

    if opt.use_arange_t and opt.train_bs_t != opt.interval:
        print('[warning] reset opt.train_bs_t to {} since use_arange_t is enabled'.format(opt.interval))
        opt.train_bs_t = opt.interval

    opt.experiment_problem_path = os.path.join(opt.experiments_path, opt.problem_name)
    os.makedirs(opt.experiment_problem_path, exist_ok=True)

    config_path = '%d_%d_%d_%.2f' % (opt.num_inner_iterations, opt.policy_updates, opt.base_discretisation, opt.var)
    opt.experiment_path = os.path.join(opt.experiment_problem_path, config_path)

    opt.ckpt_path = os.path.join(opt.experiment_path, 'checkpoints')
    os.makedirs(opt.ckpt_path, exist_ok=True)

    opt.eval_path = os.path.join(opt.experiment_path, 'eval')
    os.makedirs(opt.eval_path, exist_ok=True)

    if opt.snapshot_freq:
        os.makedirs(os.path.join(opt.eval_path, 'forward'), exist_ok=True)
        os.makedirs(os.path.join(opt.eval_path, 'backward'), exist_ok=True)

    if (opt.FID_freq and util.exist_FID_ckpt(opt)) or util.is_toy_dataset(opt):
        opt.generated_data_path = os.path.join(
            opt.eval_path, 'backward', 'generated_data'
        )
        os.makedirs(opt.generated_data_path, exist_ok=True)
    # util.check_duplication(opt)

    # ========= auto assert & (kind) warning =========
    if opt.forward_net=='ncsnpp' or opt.backward_net=='ncsnpp':
        if model_configs['ncsnpp'].training.continuous==False:
            assert opt.interval==201

    if opt.DSM_warmup:
        assert opt.train_method == 'alternate'

    if opt.load is not None:
        assert not opt.DSM_warmup, 'Already load some models, no need to DSM-warm-up!'

    if opt.train_method is not None:
        if opt.num_FID_sample>10000:
            print(util.green("[warning] you are in the training phase, are you sure you want to have large number FID evaluation?"))
        if opt.snapshot_freq<1:
            print(util.green("[warning] you are in the training phase, are you sure you do not want to have snapshot?"))

    if not opt.reuse_traj:
        print(util.green("[warning] double check that you do not want to reuse FID evaluation trajectory for training!!!"))

    # ========= print options =========
    for o in vars(opt):
        print(util.green(o),":",util.yellow(getattr(opt,o)))
    print()

    return opt
