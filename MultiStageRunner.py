
import os, time, gc

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import SGD, RMSprop, Adagrad, AdamW, lr_scheduler, Adam
from torch.utils.tensorboard import SummaryWriter
#from torch_ema import TorchEMA as ExponentialMovingAverage
from torch_ema import ExponentialMovingAverage
import torchvision
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

class EarlyStoppingCallback():
    def __init__(self, patience=5, loss_values=None):
        self.patience = patience

        if loss_values is None:
            self.loss_values = []
        else:
            self.loss_values = loss_values

        self.stopping_signal=0
    
    def add_value(self, value):
        self.loss_values.append(value)

    def __call__(self):
        if len(self.loss_values)<=2*self.patience:
            return False
        else:
            previous_values = self.loss_values[:-self.patience]
            min_from_prev_vals = min(previous_values)
            patience_values = self.loss_values[-self.patience:]
            min_from_patience_vals = min(patience_values)

            if min_from_patience_vals > min_from_prev_vals:
                return True
            else:
                return False

class MultistageEarlyStoppingCallback():
    def __init__(self, patience, loss_values):
        self.early_stopping_callbacks={}
        self.loss_values = loss_values
        for direction in loss_values.keys():
            for stage in loss_values[direction].keys():
                self.early_stopping_callbacks['%s_%s' % (direction, stage)] = EarlyStoppingCallback(patience, loss_values[direction][stage])
    
    def add_stage_value(self, value, direction, stage):
        self.early_stopping_callbacks['%s_%s' % (direction, stage)].add_value(value)

    def add_value(self, value):
        for direction in value.keys():
            for stage in value[direction].keys():
                self.early_stopping_callbacks['%s_%s' % (direction, stage)].add_value(value[direction][stage])

    def __call__(self):
        checks = []
        for callback in self.early_stopping_callbacks.keys():
            checks.append(self.early_stopping_callbacks[callback]())
        return all(checks)
    
    def get_stages_status(self):
        #returns the convergence status of all the intermediate stages
        stages = self.loss_values['forward'].keys()
        status = {}
        for stage in stages:
            forward_converged = self.early_stopping_callbacks['%s_%s' % ('forward', stage)]()
            backward_converged = self.early_stopping_callbacks['%s_%s' % ('backward', stage)]()
            if forward_converged and backward_converged:
                status[stage] = 1
            else:
                status[stage] = 0
        return status

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
        
        self.starting_outer_it = self.z_f.starting_outer_it.item()
        self.starting_inner_it = self.z_f.starting_inner_it.item()
        self.global_step = self.z_f.global_step.item()

        
        self.losses = {}
        self.losses['outer_it_%d' % self.starting_outer_it] = {}
        self.losses['outer_it_%d' % self.starting_outer_it]['forward'] = {}
        self.losses['outer_it_%d' % self.starting_outer_it]['backward'] = {}

        for i in range(1, opt.max_num_intervals//2**(self.starting_outer_it-1)+1):
            self.losses['outer_it_%d' % self.starting_outer_it]['forward'][str(i)] = getattr(self.z_f, 'outer_it_%d_forward_loss_%d' % (self.starting_outer_it, i)).tolist()
            self.losses['outer_it_%d' % self.starting_outer_it]['backward'][str(i)] = getattr(self.z_f, 'outer_it_%d_backward_loss_%d' % (self.starting_outer_it, i)).tolist()

        if opt.log_tb: # tensorboard related things
            self.it_f = 0
            self.it_b = 0

            log_dir = os.path.join(opt.experiment_path, 'logs')
            self.writer=SummaryWriter(log_dir=log_dir)

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
    def get_drift_fn(self, dyn):
        def drift_fn(x, t):
            f = dyn._f(x,t)
            g = dyn._g(t)
            z_forward = self.z_f(x,t)
            z_backward = self.z_b(x,t)
            return f+1/2*g*(z_forward-z_backward)
        return drift_fn

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

    def compute_val_loss(self, opt, inter_pq_s, discretisation):
        #validation start
        freeze_policy(self.z_f)
        freeze_policy(self.z_b)

        #*** I need to switch z_f and z_b to the ema weights.

        intervals = list(inter_pq_s.keys()).copy()
        forward_loss, backward_loss = {}, {}
        for interval_key in intervals:
            p, q = inter_pq_s[interval_key]

            interval_dyn = sde.build(opt, p, q)
            ts = torch.linspace(p.time, q.time, discretisation)
            new_dt = ts[1]-ts[0]
            interval_dyn.dt = new_dt
            ts = ts.to(opt.device)

            loss = self.compute_interval_val_loss(opt, 'forward', interval_dyn, ts)
            forward_loss[str(interval_key+1)] = loss

            loss = self.compute_interval_val_loss(opt, 'backward', interval_dyn, ts)
            backward_loss[str(interval_key+1)] = loss
        
        average_forward_loss = sum([forward_loss[key] for key in forward_loss.keys()])/len(forward_loss.keys())
        average_backward_loss = sum([backward_loss[key] for key in backward_loss.keys()])/len(backward_loss.keys())
        monitor_loss = (average_forward_loss + average_backward_loss)/2

        #validation end
        activate_policy(self.z_f)
        activate_policy(self.z_b)

        return monitor_loss

    def compute_interval_val_loss(self, opt, direction, dyn, ts):
        policy_opt, policy_impt = {
            'forward':  [self.z_f, self.z_b], # train forward,   sample from backward
            'backward': [self.z_b, self.z_f], # train backward, sample from forward
        }.get(direction)

        batch_x = opt.samp_bs
        batch_t = ts.size(0)
        xs, zs_impt, ts_ = self.sample_train_data(opt, policy_impt, dyn, ts)
        xs=util.flatten_dim01(xs)
        zs_impt=util.flatten_dim01(zs_impt)
        ts_=ts_.repeat(batch_x)
        assert xs.shape[0] == ts_.shape[0]
        assert zs_impt.shape[0] == ts_.shape[0]

        xs=xs.to(opt.device)
        zs_impt=zs_impt.to(opt.device)

        loss, zs = compute_sb_nll_alternate_train(
                opt, dyn, ts_, xs, zs_impt, policy_opt, return_z=True
            )

        assert not torch.isnan(loss)
        return loss

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

            xs, zs_impt, ts_ = self.sample_train_data(opt, policy_impt, dyn, ts)

            xs.requires_grad_(True)
            xs=util.flatten_dim01(xs)
            zs_impt=util.flatten_dim01(zs_impt)
            ts_=ts_.repeat(batch_x)
            #print(ts.shape)
            assert xs.shape[0] == ts_.shape[0]
            assert zs_impt.shape[0] == ts_.shape[0]

            # -------- compute loss and backprop --------
            xs=xs.to(opt.device)
            zs_impt=zs_impt.to(opt.device)

            loss, zs = compute_sb_nll_alternate_train(
                opt, dyn, ts_, xs, zs_impt, policy_opt, return_z=True
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

    
    def convergence_status_to_probs(self, status, k=2):
        n = len(status.keys())
        s = sum([status[key] for key in status.keys()])
        a = k / (s + k*(n-s))
        probs = {}
        for key in status.keys():
            if status[key] == 1:
                probs[key] = a/k
            else:
                probs[key] = a
        return probs

    def sb_outer_alternating_iteration(self, opt,
                                            optimizer_f, optimizer_b, 
                                            sched_f, sched_b, 
                                            inter_pq_s, discretisation, 
                                            tr_steps, outer_it):

        start_inner_it = self.starting_inner_it
        early_stopper = MultistageEarlyStoppingCallback(patience=opt.stopping_patience, loss_values=self.losses['outer_it_%d' % outer_it])
        for inner_it in tqdm(range(start_inner_it, opt.num_inner_iterations+1)):
            stop = early_stopper()
            if stop:
                break
            
            status = early_stopper.get_stages_status()
            probs = self.convergence_status_to_probs(status)
            intervals = list(probs.keys())

            if inner_it % 1000 == 0:
                print(probs)
                
            weights = [probs[key] for key in intervals]
            interval_key = random.choices(intervals, weights=weights, k=1)[0]
            
            interval_key = int(interval_key)-1
            p, q = inter_pq_s[interval_key]

            interval_dyn = sde.build(opt, p, q)
            ts = torch.linspace(p.time, q.time, discretisation)
            new_dt = ts[1]-ts[0]
            interval_dyn.dt = new_dt
            ts = ts.to(opt.device)

            losses = self.alternating_policy_update(opt, 'forward', interval_dyn, ts, tr_steps=tr_steps)
            f_loss = torch.mean(torch.tensor(losses)).item()
            self.losses['outer_it_%d' % outer_it]['forward'][str(interval_key+1)].append(f_loss)

            losses = self.alternating_policy_update(opt, 'backward', interval_dyn, ts, tr_steps=tr_steps)
            b_loss = torch.mean(torch.tensor(losses)).item()
            self.losses['outer_it_%d' % outer_it]['backward'][str(interval_key+1)].append(b_loss)

            if inner_it % opt.inner_it_save_freq == 0 and inner_it !=0:
                keys = ['z_f','optimizer_f','ema_f','z_b','optimizer_b','ema_b']
                util.multi_SBP_save_checkpoint(opt, self, keys, outer_it, inner_it)

            if inner_it % opt.sampling_freq == 0:
                with torch.no_grad():
                    sample = self.multi_sb_generate_sample(opt, inter_pq_s, discretisation)
                    sample = sample.to('cpu')
                    #gt_sample = inter_pq_s[0][0].sample()
                
                img_dataset = util.is_image_dataset(opt)
                if img_dataset:
                    sample_imgs =  sample.cpu()
                    grid_images = torchvision.utils.make_grid(sample_imgs, normalize=True, scale_each=True)
                    self.writer.add_image('samples -- outer_it:%d - inner_it:%d' % (outer_it, inner_it), grid_images, self.global_step)
                else:
                    problem_name = opt.problem_name
                        
                    p, q = inter_pq_s[0]
                    dyn = sde.build(opt, p, q)
                    img = self.tensorboard_scatter_and_quiver_plot(opt, p, dyn, sample)
                    self.writer.add_image('samples and ODE vector field, outer_it:%d - inner_it:%d' % (outer_it, inner_it), img)
            
            self.global_step += 1

            self.writer.add_scalar('outer_it_%d_forward_loss_%d' % (outer_it, (interval_key+1)), f_loss, global_step=len(self.losses['outer_it_%d' % outer_it]['forward'][str(interval_key+1)]))
            self.writer.add_scalar('outer_it_%d_backward_loss_%d' % (outer_it, (interval_key+1)), b_loss, global_step=len(self.losses['outer_it_%d' % outer_it]['backward'][str(interval_key+1)]))

            early_stopper.add_stage_value(f_loss, 'forward', str(interval_key+1))
            early_stopper.add_stage_value(b_loss, 'backward', str(interval_key+1))

            if inner_it % 100 == 50:
                self.writer.flush()

            self.z_f.starting_inner_it = torch.tensor(inner_it)
            self.starting_inner_it = inner_it
            self.z_f.global_step = torch.tensor(self.global_step)
            
            self.z_f.register_buffer('outer_it_%d_forward_loss_%d' % (outer_it, (interval_key+1)), torch.tensor(self.losses['outer_it_%d' % outer_it]['forward'][str(interval_key+1)]))
            self.z_f.register_buffer('outer_it_%d_backward_loss_%d' % (outer_it, (interval_key+1)), torch.tensor(self.losses['outer_it_%d' % outer_it]['backward'][str(interval_key+1)]))

        
        #reset after the end of the outer iteration.
        self.starting_inner_it = 1
        self.z_f.starting_inner_it = torch.tensor(self.starting_inner_it)

    def sb_alterating_train(self, opt):
        #assert not util.is_image_dataset(opt)
        policy_f, policy_b = self.z_f, self.z_b
        policy_f = activate_policy(policy_f)
        policy_b = activate_policy(policy_b)
        optimizer_f, _, sched_f = self.get_optimizer_ema_sched(policy_f)
        optimizer_b, _, sched_b = self.get_optimizer_ema_sched(policy_b)
        
        outer_iterations = self.num_outer_iterations
        num_intervals = self.max_num_intervals
        tr_steps=opt.policy_updates
        for outer_it in range(self.starting_outer_it, outer_iterations+1):
            inter_pq_s = self.setup_intermediate_distributions(opt, self.log_SNR_max, self.log_SNR_min, num_intervals)
            new_discretisation = self.compute_discretisation(opt, outer_it)
            
            #initialise the losses for the next outer iteration
            self.losses['outer_it_%d' % outer_it]={}
            self.losses['outer_it_%d' % outer_it]['forward']={}
            self.losses['outer_it_%d' % outer_it]['backward']={}
            for i in range(1, opt.max_num_intervals//2**(outer_it-1)+1):
                self.losses['outer_it_%d' % outer_it]['forward'][str(i)] = []
                self.losses['outer_it_%d' % outer_it]['backward'][str(i)] = []

            self.sb_outer_alternating_iteration(opt,
                                            optimizer_f, optimizer_b, 
                                            sched_f, sched_b, 
                                            inter_pq_s, new_discretisation, 
                                            tr_steps, outer_it)
            num_intervals = num_intervals // 2

            self.z_f.starting_outer_it += 1
    
    def compute_discretisation(self, opt, outer_it):
        if opt.discretisation_policy == 'double':
            return self.base_discretisation * 2 ** (outer_it-1)
        elif opt.discretisation_policy == 'constant':
            return self.base_discretisation
        else:
            return NotImplementedError('%s is not supported. Please implement it here.' % opt.discretisation_policy)

    def tensorboard_scatter_and_quiver_plot(self, opt, p, dyn, sample):
        drift_fn = self.get_drift_fn(dyn)
        t = torch.tensor(p.time)

        lims = {'gmm': [-17, 17], 'checkerboard': [-7, 7], 'moon-to-spiral':[-20, 20],
                }.get(opt.problem_name)
        
        xs = torch.linspace(lims[0], lims[1], 20)
        ys = torch.linspace(lims[0], lims[1], 20)
        grid_x, grid_y = torch.meshgrid([xs, ys])

        def create_mesh_points(x,y):
            z = []
            for i in range(x.size(0)):
                for j in range(y.size(0)):
                    z.append([x[i], y[j]])
            return torch.tensor(z)
        
        mesh_points = create_mesh_points(xs,ys)
        drifts = drift_fn(mesh_points.to(opt.device), t.to(opt.device)).detach().cpu()

        quiver_img = util.scatter_and_quiver(sample[:,0], sample[:,1], 
                                        mesh_points[:,0], mesh_points[:,1], 
                                        drifts[:,0], drifts[:,1])
        return quiver_img

    def tensorboard_scatter_plot(self, sample, problem_name, inner_it, outer_it):
        lims = {'gmm': [-17, 17], 'checkerboard': [-7, 7], 'moon-to-spiral':[-20, 20],
                }.get(problem_name)
        
        xlim = ylim = lims
        title = 'outer_it:%d-inner_it:%d' % (outer_it, inner_it)
        scatter_plot_tbimage = util.scatter(sample[:,0], sample[:,1], 
                                            title=title, xlim=xlim, 
                                            ylim=ylim)
        return scatter_plot_tbimage

    def scatter_plot(self, sample, problem_name, fn, save_dir):
        lims = {'gmm': [-17, 17],
                'checkerboard': [-7, 7],
                'moon-to-spiral':[-20, 20],
                }.get(problem_name)
        
        fn_pdf = os.path.join(save_dir, fn + '.png')

        plt.scatter(sample[:,0],sample[:,1], s=5)
        plt.xlim(*lims)
        plt.ylim(*lims)
        plt.savefig(fn_pdf)
        plt.clf()
    
    def group_scatter_plot(self, samples, problem_name, fn, save_dir):
        lims = {'gmm': [-17, 17],
                'checkerboard': [-7, 7],
                'moon-to-spiral':[-20, 20],
                }.get(problem_name)

        #we have a group of samples from intermediate distributions. Mainly for debugging purposes.
        fn_pdf = os.path.join(save_dir, fn + '.png')

        p_samples = samples[0]
        q_samples = samples[1]
        frames = p_samples.size(0)
        fig, axs = plt.subplots(2, frames)

        for i in range(frames):
            axs[0, i].scatter(p_samples[i][:,0], p_samples[i][:,1], s=3)
            #axs[0, i].set_xlim(*lims)
            #axs[0, i].set_ylim(*lims)
        
        for i in range(frames):
            axs[1, i].scatter(q_samples[i][:,0], q_samples[i][:,1], s=3)
            #axs[1, i].set_xlim(*lims)
            #axs[1, i].set_ylim(*lims)
        
        fig.tight_layout()
        plt.savefig(fn_pdf)
        plt.clf()

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
            
            _, _, initial_sample = interval_dyn.sample_traj(ts, self.z_b,
                                                            save_traj=False,
                                                            initial_sample=initial_sample)
        sample = initial_sample
        return initial_sample


    def experimental_features(self, opt):
        img_dataset = util.is_image_dataset(opt)
        if img_dataset:
            self.generate_samples(opt, discretisation=16, stochastic=True)
        else:
            self.visualize_trajectories(opt, discretisation=4, stochastic=False)
        

    def generate_samples(self, opt, discretisation, stochastic):
        #save_path = os.path.join(opt.experiment_path, 'testing')
        #os.makedirs(save_path, exist_ok=True)
        sampled = self.sample(opt, discretisation=discretisation, stochastic=stochastic)
        sample = sampled['sample']
        sample_imgs =  sample.cpu()
        grid_images = torchvision.utils.make_grid(sample_imgs, normalize=True, scale_each=True)

        writer=SummaryWriter(log_dir=os.path.join(opt.experiment_path, 'testing'))
        writer.add_image('test samples - discretisation:%d - %s' % (discretisation, 'stochastic' if stochastic else 'deterministic'), grid_images)
        writer.flush()
        writer.close()

    def visualize_trajectories(self, opt, discretisation=16, stochastic=True):
        save_path = os.path.join(opt.experiment_path, 'testing')
        os.makedirs(save_path, exist_ok=True)

        sampled = self.sample(opt, discretisation=discretisation, stochastic=stochastic)
        traj = sampled['trajectory']
        x = sampled['sample']

        plt.figure()
        for i in range(traj.size(0)):
            color = (np.random.rand(), np.random.rand(), np.random.rand())
            plt.plot(traj[i,1:,0], traj[i,1:,1], color=color, alpha=0.3)

        plt.scatter(x[:,0], x[:,1])
        plt.savefig(os.path.join(save_path, 'sampled-traj-%s.png' % ('sde' if stochastic else 'ode')))

        encoded = self.encode(opt, discretisation=discretisation, stochastic=stochastic)
        traj = encoded['trajectory']
        x = encoded['sample']

        plt.figure()
        for i in range(traj.size(0)):
            color = (np.random.rand(), np.random.rand(), np.random.rand())
            plt.plot(traj[i,1:,0], traj[i,1:,1], color=color, alpha=0.3)

        plt.scatter(x[:,0], x[:,1])
        plt.savefig(os.path.join(save_path, 'encoded-traj-%s.png' % ('sde' if stochastic else 'ode')))


    def estimate_average_curvature(self, opt):
        pass
    
    def estimate_discretisation(self, opt):
        #investigate the effect of discretisation on the bias of the loss estimate.
        #say we are at the converged stage of the outer iteration i where we have N_i intervals and d_i training discretisation.
        #We now want to move to the next outer iteration i where we have N_i/2 intervals and d_{i+1} discretisation.
        #A natural choice for the discretisation is 2*d_i. 
        #Even this value can lead to faster overall training 
        #because we observed that the subsequent outer iterations converge very fast 
        #compared to the first outer iteration with this discretisation.

        #We could investigate though whether coarser discretisation can be used in the next stages 
        #(given that the intermediate bridges are getting close to optimal transport maps 
        #if a small diffusion coefficient is used for the simple SDE)

        #initially we could investigate the effect of discretisation on the loss estimate.
        pass

    @torch.no_grad()
    def encode(self, opt, discretisation, save_traj=True, stochastic=True):
        #1.) detect number of SBP stages
        outer_it = self.z_f.starting_outer_it.item()
        max_num_intervals = opt.max_num_intervals
        num_intervals = max_num_intervals // 2**(outer_it-1)
        inter_pq_s = self.setup_intermediate_distributions(opt, self.log_SNR_max, self.log_SNR_min, num_intervals)

        sorted_keys = sorted(list(inter_pq_s.keys()))
        
        for i, key in tqdm(enumerate(sorted_keys)):
            
            p, q = inter_pq_s[key]
            interval_dyn = sde.build(opt, p, q)
            ts = torch.linspace(p.time, q.time, discretisation)
            new_dt = ts[1]-ts[0]
            interval_dyn.dt = new_dt
            ts = ts.to(opt.device)
            ts = torch.flip(ts,dims=[0])

            if i==0:
                x = p.sample().to(opt.device) #initialise from p_data
                xs = torch.empty((x.size(0), discretisation*num_intervals+1, *x.shape[1:])) if save_traj else None

            if save_traj:
                xs[:,0,::] = x.detach().cpu()

            for idx, t in enumerate(ts):
                if stochastic:
                    f = interval_dyn.f(x, t, direction='forward')
                    backward_policy = self.z_b
                    z = backward_policy(x,t)
                    dw = interval_dyn.dw(x)
                    g = interval_dyn.g(t)
                    dt = interval_dyn.dt
                    x = x + (f - g*z)*(dt) + g*dw
                else: #ODE case
                    drift_fn = self.get_drift_fn(interval_dyn)
                    drift = drift_fn(x,t)
                    x = x + drift * (interval_dyn.dt)
                
                
                if save_traj:
                    xs[:,i*discretisation+idx+1,::] = x.detach().cpu()
            
        return {'trajectory': xs, 'sample':x.detach().cpu()}

    @torch.no_grad()
    def sample(self, opt, discretisation, save_traj=True, stochastic=True):
        #1.) detect number of SBP stages
        outer_it = self.z_f.starting_outer_it.item()
        max_num_intervals = opt.max_num_intervals
        num_intervals = max_num_intervals // 2**(outer_it-1)
        inter_pq_s = self.setup_intermediate_distributions(opt, self.log_SNR_max, self.log_SNR_min, num_intervals)

        sorted_keys = sorted(list(inter_pq_s.keys()), reverse=True)
        
        for i, key in tqdm(enumerate(sorted_keys)):
            
            p, q = inter_pq_s[key]
            interval_dyn = sde.build(opt, p, q)
            ts = torch.linspace(p.time, q.time, discretisation)
            new_dt = ts[1]-ts[0]
            interval_dyn.dt = new_dt
            ts = ts.to(opt.device)
            ts = torch.flip(ts,dims=[0])

            if i==0:
                x = q.sample() #initialise from p_prior
                xs = torch.empty((x.size(0), discretisation*num_intervals+1, *x.shape[1:])) if save_traj else None

            if save_traj:
                xs[:,0,::] = x.detach().cpu()

            for idx, t in enumerate(ts):
                if stochastic:
                    f = interval_dyn.f(x, t, direction='forward')
                    backward_policy = self.z_b
                    z = backward_policy(x,t)
                    dw = interval_dyn.dw(x)
                    g = interval_dyn.g(t)
                    dt = interval_dyn.dt
                    x = x + (f - g*z)*(-dt) + g*dw
                else: #ODE case
                    drift_fn = self.get_drift_fn(interval_dyn)
                    drift = drift_fn(x,t)
                    x = x + drift * (-interval_dyn.dt)
                
                
                if save_traj:
                    xs[:,i*discretisation+idx+1,::] = x.detach().cpu()
            
        return {'trajectory': xs, 'sample':x.detach().cpu()}

    def sanity_check(self, opt, sanity_check_type = 'marginals'):
        if sanity_check_type == 'marginals':
            #we check whether the marginal distributions are the correct ones.
            num_intervals=4
            intermediate_distributions = self.setup_intermediate_distributions(opt, self.log_SNR_max, self.log_SNR_min, num_intervals)
            
            p_samples, q_samples = [], []
            for i in range(num_intervals):
                p, q = intermediate_distributions[i]
                p_sample = p.sample().to('cpu')
                p_samples.append(p_sample)
                q_sample = q.sample().to('cpu')
                q_samples.append(q_sample)
            
            p_samples = torch.stack(p_samples)
            q_samples = torch.stack(q_samples)
            stack_samples = torch.stack([p_samples, q_samples])
            fn = 'sanity-check-perturbation'
            
            save_dir = os.path.join(opt.experiment_path, 'debug')
            os.makedirs(save_dir, exist_ok=True)

            self.group_scatter_plot(stack_samples, opt.problem_name, fn, save_dir)

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

        if inner_it % 500 == 0:
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
