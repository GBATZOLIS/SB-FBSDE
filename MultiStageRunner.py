
import os, time, gc

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import SGD, RMSprop, Adagrad, AdamW, lr_scheduler, Adam
from torch.utils.tensorboard import SummaryWriter
from torch_ema import TorchEMA as ExponentialMovingAverage

import policy
import sde
from loss import compute_sb_nll_alternate_train, compute_sb_nll_joint_train
import data
import util

from ipdb import set_trace as debug
import random

def build_optimizer_ema_sched(opt, policy):
    direction = policy.direction

    optim_name = {
        'Adam': Adam,
        'AdamW': AdamW,
        'Adagrad': Adagrad,
        'RMSprop': RMSprop,
        'SGD': SGD,
    }.get(opt.optimizer)

    optim_dict = {
            "lr": opt.lr_f if direction=='forward' else opt.lr_b,
            'weight_decay':opt.l2_norm,
    }
    if opt.optimizer == 'SGD':
        optim_dict['momentum'] = 0.9

    optimizer = optim_name(policy.parameters(), **optim_dict)
    ema = ExponentialMovingAverage(policy.parameters(), decay=0.99)
    if opt.lr_gamma < 1.0:
        sched = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_gamma)
    else:
        sched = None

    return optimizer, ema, sched

def freeze_policy(policy):
    for p in policy.parameters():
        p.requires_grad = False
    policy.eval()
    return policy

def activate_policy(policy):
    for p in policy.parameters():
        p.requires_grad = True
    policy.train()
    return policy

class MultiStageRunner():
    def __init__(self, opt):
        super(MultiStageRunner,self).__init__()
        self.start_time = time.time()

        #multistage settings
        self.log_SNR_max = opt.log_SNR_max
        self.log_SNR_min = opt.log_SNR_min
        
        self.num_outer_iterations = opt.num_outer_iterations
        self.max_num_intervals = opt.max_num_intervals
        self.base_discretisation = opt.base_discretisation

        # build boundary distribution (p: target, q: prior)
        self.p, self.q = data.build_boundary_distribution(opt)
        # build dynamics, forward (z_f) and backward (z_b) policies
        self.dyn = sde.build(opt, self.p, self.q)
        self.z_f = policy.build(opt, self.dyn, 'forward')  # p -> q
        self.z_b = policy.build(opt, self.dyn, 'backward') # q -> p
        self.optimizer_f, self.ema_f, self.sched_f = build_optimizer_ema_sched(opt, self.z_f)
        self.optimizer_b, self.ema_b, self.sched_b = build_optimizer_ema_sched(opt, self.z_b)

        if opt.load:
            util.restore_checkpoint(opt, self, opt.load)

        if opt.log_tb: # tensorboard related things
            self.it_f = 0
            self.it_b = 0
            self.writer=SummaryWriter(
                log_dir=os.path.join('runs', opt.log_fn) if opt.log_fn is not None else None
            )

    def setup_intermediate_distributions(self, opt, log_SNR_max, log_SNR_min, num_intervals):
        #return {interval_number:[p,q]}
        snr_vals = np.logspace(log_SNR_max, log_SNR_min, num=num_intervals+1)
        times = torch.linspace(opt.t0, opt.T, num_intervals+1)

        intermediate_distributions = {}
        for i in range(num_intervals):
            if i < num_intervals - 1:
                p = data.build_perturbed_data_sampler(opt, opt.samp_bs, snr_vals[i])
                q = data.build_perturbed_data_sampler(opt, opt.samp_bs, snr_vals[i+1])
            elif i == num_intervals - 1:
                p = data.build_perturbed_data_sampler(opt, opt.samp_bs, snr_vals[i])
                q = data.build_prior_sampler(opt, opt.samp_bs)
                
            p.time = times[i]
            q.time = times[i+1]
            intermediate_distributions[i] = [p, q]
        
        return intermediate_distributions

    def get_optimizer_ema_sched(self, z):
        if z == self.z_f:
            return self.optimizer_f, self.ema_f, self.sched_f
        elif z == self.z_b:
            return self.optimizer_b, self.ema_b, self.sched_b
        else:
            raise RuntimeError()

    @torch.no_grad()
    def sample_train_data(self, opt, policy_impt, dyn, ts):
        _, ema_impt, _ = self.get_optimizer_ema_sched(policy_impt)
        with ema_impt.average_parameters():
            policy_impt = freeze_policy(policy_impt)
            xs, zs, _ = dyn.sample_traj(ts, policy_impt, corrector=None)

        #print('generate train data from [{}]!'.format(util.red('sampling')))
        assert xs.shape[0] == opt.samp_bs
        assert xs.shape[1] == len(ts)
        assert xs.shape == zs.shape
        gc.collect()
        return xs, zs, ts


    def alternating_policy_update(self, direction, dyn, ts, tr_steps=1):
        policy_opt, policy_impt = {
            'forward':  [self.z_f, self.z_b], # train forward,   sample from backward
            'backward': [self.z_b, self.z_f], # train backward, sample from forward
        }.get(direction)

        policy_impt = freeze_policy(policy_impt)
        policy_opt = activate_policy(policy_opt)

        optimizer, ema, sched = self.get_optimizer_ema_sched(policy_opt)

        batch_x = opt.samp_bs
        for i in range(tr_steps):
            optimizer.zero_grad()

            xs, zs_impt, ts = self.sample_train_data(opt, policy_impt, dyn, ts)

            # -------- compute loss and backprop --------
            loss, zs = compute_sb_nll_alternate_train(
                opt, dyn, ts, xs, zs_impt, policy_opt, return_z=True
            )
            assert not torch.isnan(loss)

            loss.backward()
            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm(policy_opt.parameters(), opt.grad_clip)
            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            # -------- logging --------
            #zs = util.unflatten_dim01(zs, [len(samp_x_idx), len(samp_t_idx)])
            #zs_impt = zs_impt.reshape(zs.shape)
            #self.log_sb_alternate_train(
            #    opt, it, ep, stage, loss, zs, zs_impt, optimizer, direction, num_epoch
            #)


    def sb_outer_alternating_iteration(self, opt,
                                            optimizer_f, optimizer_b, 
                                            sched_f, sched_b, 
                                            inter_pq_s, discretisation):

        for it in range(opt.num_itr):
            interval_key = random.choice(list(inter_pq_s.keys()))
            p, q = inter_pq_s[interval_key]

            interval_dyn = sde.build(opt, p, q)
            ts = torch.linspace(p.time, q.time, discretisation)
            new_dt = ts[1]-ts[0]
            interval_dyn.dt = new_dt

            self.alternating_policy_update('forward', interval_dyn, ts, tr_steps=1)
            self.alternating_policy_update('backward', interval_dyn, ts, tr_steps=1)


    def sb_alterating_train(self, opt):
        assert not util.is_image_dataset(opt)
        policy_f, policy_b = self.z_f, self.z_b
        policy_f = activate_policy(policy_f)
        policy_b = activate_policy(policy_b)
        optimizer_f, _, sched_f = self.get_optimizer_ema_sched(policy_f)
        optimizer_b, _, sched_b = self.get_optimizer_ema_sched(policy_b)
        
        outer_iterations = self.num_outer_iterations
        num_intervals = self.max_num_intervals
        for out_it in range(outer_iterations):
            inter_pq_s = self.setup_intermediate_distributions(opt, self.log_SNR_max, self.log_SNR_min, num_intervals)
            self.sb_outer_alternating_iteration(opt,
                                            optimizer_f, optimizer_b, 
                                            sched_f, sched_b, 
                                            inter_pq_s, self.base_discretisation * 2 ** out_it)
            num_intervals = num_intervals // 2


    def sb_joint_train(self, opt):
        assert not util.is_image_dataset(opt)
        policy_f, policy_b = self.z_f, self.z_b
        policy_f = activate_policy(policy_f)
        policy_b = activate_policy(policy_b)
        optimizer_f, _, sched_f = self.get_optimizer_ema_sched(policy_f)
        optimizer_b, _, sched_b = self.get_optimizer_ema_sched(policy_b)

        outer_iterations = self.num_outer_iterations
        num_intervals = self.max_num_intervals
        for out_it in range(outer_iterations):
            inter_pq_s = self.setup_intermediate_distributions(opt, self.log_SNR_max, self.log_SNR_min, num_intervals)
            self.sb_outer_joint_iteration(opt, policy_f, policy_b, 
                                               optimizer_f, optimizer_b, 
                                               sched_f, sched_b,
                                               inter_pq_s, self.base_discretisation * 2 ** out_it)
            num_intervals = num_intervals // 2

            #marks the end of the outer iteration. We should have a solution of the intermediate num_intervals SBPs.
            #when num_intervals=1, we should have the solution of the target SBP problem.


    def sb_outer_joint_iteration(self, opt, 
                                       policy_f, policy_b, 
                                       optimizer_f, optimizer_b, 
                                       sched_f, sched_b,
                                       inter_pq_s, discretisation):

        num_intervals = len(inter_pq_s.keys())

        batch_x = opt.samp_bs
        for it in range(opt.num_itr):

            optimizer_f.zero_grad()
            optimizer_b.zero_grad()

            interval_key = random.choice(list(inter_pq_s.keys()))
            p, q = inter_pq_s[interval_key]

            interval_dyn = sde.build(opt, p, q)
            ts = torch.linspace(p.time, q.time, discretisation)
            new_dt = ts[1]-ts[0]
            interval_dyn.dt = new_dt

            xs_f, zs_f, x_term_f = interval_dyn.sample_traj(ts, policy_f, save_traj=True)
            xs_f = util.flatten_dim01(xs_f)
            zs_f = util.flatten_dim01(zs_f)
            _ts = ts.repeat(batch_x)

            loss = compute_sb_nll_joint_train(
                opt, batch_x, interval_dyn, _ts, xs_f, zs_f, x_term_f, policy_b
            )
            loss.backward()

            optimizer_f.step()
            optimizer_b.step()

            if sched_f is not None: sched_f.step()
            if sched_b is not None: sched_b.step()

            self.log_sb_joint_train(opt, it, loss, optimizer_f, opt.num_itr)

            # evaluate
            if (it+1) % opt.eval_itr==0:
                with torch.no_grad():
                    xs_b, _, _ = self.dyn.sample_traj(ts, policy_b, save_traj=True)
                util.save_toy_npy_traj(opt, 'train_it{}'.format(it+1), xs_b.detach().cpu().numpy())

    def log_sb_joint_train(self, opt, it, loss, optimizer, num_itr):
        self._print_train_itr(it, loss, optimizer, num_itr, name='SB joint')
        if opt.log_tb:
            step = self.update_count('backward')
            self.log_tb(step, loss.detach(), 'loss', 'SB_joint')

    def _print_train_itr(self, it, loss, optimizer, num_itr, name):
        time_elapsed = util.get_time(time.time()-self.start_time)
        lr = optimizer.param_groups[0]['lr']
        print("[{0}] train_it {1}/{2} | lr:{3} | loss:{4} | time:{5}"
            .format(
                util.magenta(name),
                util.cyan("{}".format(1+it)),
                num_itr,
                util.yellow("{:.2e}".format(lr)),
                util.red("{:.4f}".format(loss.item())),
                util.green("{0}:{1:02d}:{2:05.2f}".format(*time_elapsed)),
        ))
    
    def update_count(self, direction):
        if direction == 'forward':
            self.it_f += 1
            return self.it_f
        elif direction == 'backward':
            self.it_b += 1
            return self.it_b
        else:
            raise RuntimeError()
    
    def log_tb(self, step, val, name, tag):
        self.writer.add_scalar(os.path.join(tag,name), val, global_step=step)
