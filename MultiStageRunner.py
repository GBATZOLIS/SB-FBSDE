
import os, time, gc

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import SGD, RMSprop, Adagrad, AdamW, lr_scheduler, Adam
from torch.utils.tensorboard import SummaryWriter
#from torch_ema import TorchEMA as ExponentialMovingAverage
from torch_ema import ExponentialMovingAverage

import policy
import sde
from loss import compute_sb_nll_alternate_train, compute_sb_nll_joint_train
import data
import util

from ipdb import set_trace as debug
import random
from tqdm import tqdm

import matplotlib.pyplot as plt


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
        snr_vals = np.logspace(log_SNR_max, log_SNR_min, num=num_intervals+1, base=np.exp(1))
        #print(snr_vals)
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


    def alternating_policy_update(self, opt, direction, dyn, ts, tr_steps=1):
        policy_opt, policy_impt = {
            'forward':  [self.z_f, self.z_b], # train forward,   sample from backward
            'backward': [self.z_b, self.z_f], # train backward, sample from forward
        }.get(direction)

        policy_impt = freeze_policy(policy_impt)
        policy_opt = activate_policy(policy_opt)

        optimizer, ema, sched = self.get_optimizer_ema_sched(policy_opt)

        batch_x = opt.samp_bs
        batch_t = ts.size(0)
        losses = []
        for it in range(tr_steps):
            optimizer.zero_grad()

            xs, zs_impt, ts = self.sample_train_data(opt, policy_impt, dyn, ts)

            xs.requires_grad_(True)
            xs=util.flatten_dim01(xs)
            zs_impt=util.flatten_dim01(zs_impt)
            ts=ts.repeat(batch_x)
            #print(ts.shape)
            assert xs.shape[0] == ts.shape[0]
            assert zs_impt.shape[0] == ts.shape[0]

            # -------- compute loss and backprop --------
            xs=xs.to(opt.device)
            zs_impt=zs_impt.to(opt.device)

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

            losses.append(loss)

        return losses


    def sb_outer_alternating_iteration(self, opt,
                                            optimizer_f, optimizer_b, 
                                            sched_f, sched_b, 
                                            inter_pq_s, discretisation, 
                                            tr_steps, outer_it):

        for inner_it in range(1, opt.num_inner_iterations+1):
            interval_key = random.choice(list(inter_pq_s.keys()))
            p, q = inter_pq_s[interval_key]

            interval_dyn = sde.build(opt, p, q)
            ts = torch.linspace(p.time, q.time, discretisation)
            new_dt = ts[1]-ts[0]
            interval_dyn.dt = new_dt
            ts = ts.to(opt.device)

            losses = self.alternating_policy_update(opt, 'forward', interval_dyn, ts, tr_steps=tr_steps)
            self.log_multistage_sb_alternate_train(opt, outer_it, inner_it, 'forward', losses)
            
            losses = self.alternating_policy_update(opt, 'backward', interval_dyn, ts, tr_steps=tr_steps)
            self.log_multistage_sb_alternate_train(opt, outer_it, inner_it, 'backward', losses)

            
            if inner_it % 25 == 0:
                with torch.no_grad():
                    sample = self.multi_sb_generate_sample(opt, inter_pq_s, discretisation)
                    sample = sample.to('cpu')
                    print(sample.size())
                
                lims = {
                        'gmm': [-17, 17],
                        'checkerboard': [-7, 7],
                        'moon-to-spiral':[-20, 20],
                    }.get(opt.problem_name)

                fn = 'outer_it:%d-inner_it:%d'% (outer_it, inner_it)
                fn_pdf = os.path.join('results', opt.dir, fn+'.pdf')

                plt.scatter(sample[:,0],sample[:,1], s=5)
                plt.xlim(*lims)
                plt.ylim(*lims)
                plt.savefig(fn_pdf)
                plt.clf()
                #util.save_toy_npy_traj(opt, 'outer_it:%d-inner_it:%d'% (outer_it, inner_it), sample.detach().cpu().numpy())

    @torch.no_grad()
    def multi_sb_generate_sample(self, opt, inter_pq_s, discretisation):
        sorted_keys = sorted(list(inter_pq_s.keys()), reverse=True)
        for i, key in tqdm(enumerate(sorted_keys)):
            p, q = inter_pq_s[key]
            interval_dyn = sde.build(opt, p, q)
            ts = torch.linspace(p.time, q.time, discretisation)
            new_dt = ts[1]-ts[0]
            interval_dyn.dt = new_dt
            ts = ts.to(opt.device)
            if i==0:
                initial_sample=None
                print('From (t,logSNR):(%.3f,-infty) to (t,logSNR):(%.3f,%.3f)' % (q.time, p.time, np.log(p.snr)))
            else:
                print('From (t,logSNR):(%.3f,%.3f) to (t,logSNR):(%.3f,%.3f)' % (q.time, np.log(q.snr), p.time, np.log(p.snr)))
            
            _, _, initial_sample = interval_dyn.sample_traj(ts, self.z_b,
                                                            save_traj=False,
                                                            initial_sample=initial_sample)

        sample = initial_sample
        return initial_sample


    def sb_alterating_train(self, opt):
        assert not util.is_image_dataset(opt)
        policy_f, policy_b = self.z_f, self.z_b
        policy_f = activate_policy(policy_f)
        policy_b = activate_policy(policy_b)
        optimizer_f, _, sched_f = self.get_optimizer_ema_sched(policy_f)
        optimizer_b, _, sched_b = self.get_optimizer_ema_sched(policy_b)
        
        outer_iterations = self.num_outer_iterations
        num_intervals = self.max_num_intervals
        tr_steps=opt.policy_updates
        for outer_it in range(1, outer_iterations+1):
            inter_pq_s = self.setup_intermediate_distributions(opt, self.log_SNR_max, self.log_SNR_min, num_intervals)
            self.sb_outer_alternating_iteration(opt,
                                            optimizer_f, optimizer_b, 
                                            sched_f, sched_b, 
                                            inter_pq_s, self.base_discretisation * 2 ** (outer_it-1), 
                                            tr_steps, outer_it)
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

    def log_multistage_sb_alternate_train(self, opt, outer_it, inner_it, direction, losses):
        time_elapsed = util.get_time(time.time()-self.start_time)
        avg_loss = torch.mean(torch.tensor(losses)).item()
        update_steps = len(losses)

        if inner_it % 10 == 0:
            print("direction:[{0}]| outer it:{1}/{2} | inner it:{3}/{4} | update steps: {5} | loss:{6} | time:{7} ".format(
                util.magenta("SB {}".format(direction)),
                util.cyan("{}".format(outer_it)),
                util.cyan("{}".format(opt.num_outer_iterations)),
                util.cyan("{}".format(inner_it)),
                util.cyan("{}".format(opt.num_inner_iterations)),
                util.green("{}".format(update_steps)),
                util.red("{}".format(avg_loss)),
                util.green("{0}:{1:02d}:{2:05.2f}".format(*time_elapsed))
            ))
        
        if opt.log_tb:
            step = self.update_count(direction)
            self.log_tb(step, avg_loss, 'loss', 'SB_'+direction) # SB surrogate loss (see Eq 18 & 19 in the paper)


    def log_sb_alternate_train(self, opt, it, ep, stage, loss, zs, zs_impt, optimizer, direction, num_epoch):
        time_elapsed = util.get_time(time.time()-self.start_time)
        lr = optimizer.param_groups[0]['lr']
        print("[{0}] stage {1}/{2} | ep {3}/{4} | train_it {5}/{6} | lr:{7} | loss:{8} | time:{9}"
            .format(
                util.magenta("SB {}".format(direction)),
                util.cyan("{}".format(1+stage)),
                opt.num_stage,
                util.cyan("{}".format(1+ep)),
                num_epoch,
                util.cyan("{}".format(1+it+opt.num_itr*ep)),
                opt.num_itr*num_epoch,
                util.yellow("{:.2e}".format(lr)),
                util.red("{:+.4f}".format(loss.item())),
                util.green("{0}:{1:02d}:{2:05.2f}".format(*time_elapsed)),
        ))
        if opt.log_tb:
            step = self.update_count(direction)
            neg_elbo = loss + util.compute_z_norm(zs_impt, self.dyn.dt)
            self.log_tb(step, loss.detach(), 'loss', 'SB_'+direction) # SB surrogate loss (see Eq 18 & 19 in the paper)
            self.log_tb(step, neg_elbo.detach(), 'neg_elbo', 'SB_'+direction) # negative ELBO (see Eq 16 in the paper)
            # if direction == 'forward':
            #     z_norm = util.compute_z_norm(zs, self.dyn.dt)
            #     self.log_tb(step, z_norm.detach(), 'z_norm', 'SB_forward')

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
