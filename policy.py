
import torch
import sde
import util

from ipdb import set_trace as debug

def build(opt, dyn, direction):
    print(util.magenta("build {} policy...".format(direction)))

    net_name = getattr(opt, direction+'_net')
    net = _build_net(opt, net_name)
    use_t_idx = (net_name in ['toy', 'Unet', 'DGLSB']) # t_idx is handled internally in ncsnpp
    scale_by_g = (net_name in ['ncsnpp'])

    if opt.training_scheme == 'divideNconquer':
        policy = MultiStageSchrodingerBridgePolicy(
        opt, direction, dyn, net, use_t_idx=use_t_idx, scale_by_g=scale_by_g)
    else:
        policy = SchrodingerBridgePolicy(
            opt, direction, dyn, net, use_t_idx=use_t_idx, scale_by_g=scale_by_g
        )

    print(util.red('number of parameters is {}'.format(util.count_parameters(policy))))
    policy.to(opt.device)

    return policy

def _build_net(opt, net_name):
    compute_sigma = lambda t: sde.compute_sigmas(t, opt.sigma_min, opt.sigma_max)
    zero_out_last_layer = opt.DSM_warmup

    if net_name == 'toy':
        assert util.is_toy_dataset(opt)
        from models.toy_model.Toy import build_toy
        net = build_toy(zero_out_last_layer)
    elif net_name == 'Unet':
        from models.Unet.Unet import build_unet
        net = build_unet(opt.model_configs[net_name], zero_out_last_layer)
    elif net_name == 'ncsnpp':
        from models.ncsnpp.ncsnpp import build_ncsnpp
        net = build_ncsnpp(opt.model_configs[net_name], compute_sigma, zero_out_last_layer)
    elif net_name == 'DGLSB':
        from models.DGLSB.dglsb import build_dglsb
        net = build_dglsb(zero_out_last_layer)
    else:
        raise RuntimeError()
    return net

class SchrodingerBridgePolicy(torch.nn.Module):
    # note: scale_by_g matters only for pre-trained model
    def __init__(self, opt, direction, dyn, net, use_t_idx=False, scale_by_g=True):
        super(SchrodingerBridgePolicy,self).__init__()
        self.opt = opt
        self.direction = direction
        self.dyn = dyn
        self.net = net
        self.use_t_idx = use_t_idx
        self.scale_by_g = scale_by_g

    @ property
    def zero_out_last_layer(self):
        return self.net.zero_out_last_layer


    def forward(self, x, t):
        # make sure t.shape = [batch]
        t = t.squeeze()
        if t.dim()==0: t = t.repeat(x.shape[0])
        assert t.dim()==1 and t.shape[0] == x.shape[0]

        if self.use_t_idx:
            t = t / self.opt.T * self.opt.interval

        out = self.net(x, t)

        # if the SB policy behaves as "Z" in FBSDE system,
        # the output should be scaled by the diffusion coefficient "g".
        if self.scale_by_g:
            g = self.dyn.g(t)
            g = g.reshape(x.shape[0], *([1,]*(x.dim()-1)))
            out = out * g

        return out

class MultiStageSchrodingerBridgePolicy(SchrodingerBridgePolicy):
    # note: scale_by_g matters only for pre-trained model
    def __init__(self, opt, direction, dyn, net, use_t_idx=False, scale_by_g=True):
        super(MultiStageSchrodingerBridgePolicy, self).__init__(opt, direction, dyn, net, use_t_idx, scale_by_g)
        num_intervals = opt.prev_reduction_levels // opt.reduction_levels
        self.initialize_logs(num_intervals, opt.reduction_levels)

    def initialize_logs(self, num_intervals:int, reduction_levels:int):
        self.register_buffer('num_intervals', torch.tensor(num_intervals, dtype=torch.int32))
        self.register_buffer('reduction_levels', torch.tensor(reduction_levels, dtype=torch.int32))
        self.register_buffer('starting_outer_it', torch.tensor(1, dtype=torch.int32))
        self.register_buffer('starting_inner_it', torch.tensor(1, dtype=torch.int32))
        self.register_buffer('global_step', torch.tensor(0, dtype=torch.int32))
        for i in range(1, num_intervals+1):
            self.register_buffer('outer_it_1_train_forward_loss_%d' % i, torch.tensor([]))
            self.register_buffer('outer_it_1_train_backward_loss_%d' % i, torch.tensor([]))
            self.register_buffer('outer_it_1_val_forward_loss_%d' % i, torch.tensor([]))
            self.register_buffer('outer_it_1_val_backward_loss_%d' % i, torch.tensor([]))