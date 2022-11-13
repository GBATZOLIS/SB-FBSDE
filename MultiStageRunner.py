
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
        for interval in loss_values.keys():
            self.early_stopping_callbacks['%d' % interval] = EarlyStoppingCallback(patience, loss_values[interval])
    
    def add_stage_value(self, value, interval):
        self.early_stopping_callbacks['%d' % interval].add_value(value)

    def __call__(self):
        checks = []
        for callback in self.early_stopping_callbacks.keys():
            checks.append(self.early_stopping_callbacks[callback]())
        return all(checks)
    
    def get_stages_status(self):
        #returns the convergence status of all the intermediate stages
        intervals = self.loss_values.keys()
        status = {}
        for interval in intervals:
            converged = self.early_stopping_callbacks['%d' % interval]()
            if converged:
                status[interval] = 1
            else:
                status[interval] = 0
        return status

def initialise_logs(num_intervals:int, reduction_levels:int):
    logs = {}
    logs['num_intervals'] = num_intervals
    logs['reduction_levels'] = reduction_levels
    logs['resume_info'] = {}
    logs['loss'] = {}
    for direction in ['forward', 'backward']:
        logs['resume_info'][direction] = {}
        logs['resume_info'][direction]['starting_outer_it'] = 1
        logs['resume_info'][direction]['starting_stage'] = 1
        logs['resume_info'][direction]['starting_inner_it'] = 1
        logs['resume_info'][direction]['global_step'] = 0 #probably redundant
        logs['loss'][direction] = {}
        outer_it = 1
        logs['loss'][direction][outer_it] = {}
        stage = 1
        logs['loss'][direction][outer_it][stage]={}
        for phase in ['train', 'val']:
            logs['loss'][direction][outer_it][stage][phase] = {}
            for i in range(1, num_intervals+1):
                logs['loss'][direction][outer_it][stage][phase][i] = []
    return logs

#create a new class that inherits the class of all models in reduced level j and can perform sampling.
class MultistageCombiner():
    def __init__(self, opt):
        self.multistage_model = {}
        self.opts = {}
        for i in range(1, opt.reduction_levels+1):
            opt.level_id = i
            checkpoints_path = os.path.join(opt.experiment_path, 'reduction_%d' % opt.reduction_levels, '%d' % i, 'checkpoints')
            opt.load = os.path.join(checkpoints_path, opt.reduced_models_load[i-1]+'.npz')
            self.opts[i] = opt

            print(self.opts[i].level_id)
            print(self.opts[i].load)
            self.multistage_model[i] = MultiStageRunner(self.opts[i])
        
        log_dir = os.path.join(opt.experiment_path,  'reduction_%d' % opt.reduction_levels, 'samples')
        self.writer = SummaryWriter(log_dir=log_dir)
    
    def sample(self, save_traj=True, stochastic=True):
        levels = sorted(list(self.opts.keys()), reverse=True)
        x = None
        for level in levels:
            opt = self.opts[level]
            out = self.multistage_model[level].sample(opt, opt.base_discretisation, x, save_traj, stochastic)
            x = out['sample']
        
        return x
    
    def generate_samples(self, N=1, save_traj=True, stochastic=True):
        for i in range(1, N+1):
            x = self.sample(save_traj, stochastic)
            print(x)
            self.save_sample(x, i)
        
        self.writer.close()
    
    def save_sample(self, x, i):
        opt = self.opts[1]
        img_dataset = util.is_image_dataset(opt)
        if img_dataset:
            x = x.cpu()
            grid_images = torchvision.utils.make_grid(x, nrow=int(np.sqrt(x.size(0))), normalize=True, scale_each=True)
            self.writer.add_image('%d' % i, grid_images)
            #self.writer.flush()
        else:
            x = x.cpu()
            img = self.multistage_model[1].tensorboard_scatter_plot(x, opt.problem_name, -1, -1)
            self.writer.add_image('%d' % i, img)
            #self.writer.flush()


#instruction to myself: Modify this class so that it does the training and the sampling of the reduced unit i which is found in the reduced level j.
#we need to calculate the maximum and minimum SNRs of reduced unit i. We need to understand if it is the last unit 
#(in this case the last distribution is the prior and we need to modify the sample generating method)
class MultiStageRunner():
    def __init__(self, opt):
        super(MultiStageRunner, self).__init__()
        self.start_time = time.time()

        #multistage settings
        #original SBP problem settings
        self.log_SNR_max = opt.log_SNR_max
        self.log_SNR_min = opt.log_SNR_min
        self.base_discretisation = opt.base_discretisation

        #level settings
        self.level = opt.level_id #1,2,...,N
        self.reduction_levels = opt.reduction_levels #N

        if opt.use_last_level:
            self.last_level = True if self.level == self.reduction_levels else False
        else:
            self.last_level = False

        snr_vals = np.logspace(self.log_SNR_max, self.log_SNR_min, num=self.reduction_levels+1, base=np.exp(1))
        self.level_log_SNR_max = np.log(snr_vals[self.level-1])
        self.level_log_SNR_min = np.log(snr_vals[self.level])

        print('max SNR: %.3f - min SNR: %.3f' % (self.level_log_SNR_max, self.level_log_SNR_min))

        times = torch.linspace(opt.t0, opt.T, self.reduction_levels+1)
        self.level_min_time = times[self.level-1]
        self.level_max_time = times[self.level]
        print('min time: %.3f - max time: %.3f' % (self.level_min_time, self.level_max_time))

        self.max_num_intervals = opt.prev_reduction_levels // opt.reduction_levels
        print('Max number of intervals: %d' % self.max_num_intervals)
        self.num_outer_iterations = int(np.log2(self.max_num_intervals))+1
        print('Number of outer iterations: %d' % self.num_outer_iterations)

        # build boundary distribution (p: target, q: prior)
        self.p, self.q = data.build_boundary_distribution(opt)
        # build dynamics, forward (z_f) and backward (z_b) policies
        self.dyn = sde.build(opt, self.p, self.q)
        self.z_f = policy.build(opt, self.dyn, 'forward')  # p -> q
        self.z_b = policy.build(opt, self.dyn, 'backward') # q -> p
        self.optimizer_f, self.ema_f, self.sched_f = build_optimizer_ema_sched(opt, self.z_f)
        self.optimizer_b, self.ema_b, self.sched_b = build_optimizer_ema_sched(opt, self.z_b)

        if opt.load:
            checkpoint_path = os.path.join(opt.ckpt_path, opt.load + '.npz')
            util.restore_checkpoint(opt, self, checkpoint_path)

            if self.reduction_levels < self.z_f.reduction_levels.item():
                #in this case we enter a new reduction cycle
                #we need to initialize the logs
                print('New option reduction levels less than loaded reduction levels.')
                print('We are entering a new reduction cycle.')
                self.logs = initialise_logs(self.max_num_intervals, self.reduction_levels)
            else:
                print('New option reduction levels same as the loaded reduction levels')
                print('We are either resuming training or using the loaded model for sampling.')
                logs_path = os.path.join(opt.logs_path, opt.load + '.pkl')
                self.logs = util.restore_logs(logs_path)
        else:
            self.logs = initialise_logs(self.max_num_intervals, self.reduction_levels)
        
        #every should be read/written from/to self.logs from now on.
        self.starting_outer_it = self.logs['resume_info']['forward']['starting_outer_it']
        self.starting_stage = self.logs['resume_info']['forward']['starting_stage']
        self.skip_backward = True if self.logs['resume_info']['backward']['starting_stage'] > self.logs['resume_info']['forward']['starting_stage'] else False
        self.global_step = self.logs['resume_info']['forward']['global_step']
        self.starting_inner_it = {'forward': self.logs['resume_info']['forward']['starting_inner_it'],
                                  'backward': self.logs['resume_info']['backward']['starting_inner_it']}

        self.losses = self.logs['loss']
        print(self.losses.keys())

        if opt.log_tb: # tensorboard related things
            self.it_f = 0
            self.it_b = 0

            log_dir = os.path.join(opt.experiment_path,  'reduction_%d' % opt.reduction_levels, '%d' % opt.level_id, 'tensorboard_logs')
            self.writer = SummaryWriter(log_dir=log_dir)

    #done
    def setup_intermediate_distributions(self, opt, log_SNR_max, log_SNR_min, 
                                                    min_time, max_time, 
                                                    num_intervals, discretisation_policy='constant', outer_it=1, phase='train'):
        #return {interval_number:[p,q]}
        snr_vals = np.logspace(log_SNR_max, log_SNR_min, num=num_intervals+1, base=np.exp(1))
        times = torch.linspace(min_time, max_time, num_intervals+1)

        if discretisation_policy == 'constant':
            batchsize = opt.samp_bs
        elif discretisation_policy == 'double':
            batchsize = opt.samp_bs // 2**(outer_it-1)

        intermediate_distributions = {}
        for i in range(num_intervals):
            if self.last_level:
                if i < num_intervals - 1:
                    p = data.build_perturbed_data_sampler(opt, batchsize, snr_vals[i], phase)
                    q = data.build_perturbed_data_sampler(opt, batchsize, snr_vals[i+1], phase)
                elif i == num_intervals - 1:
                    p = data.build_perturbed_data_sampler(opt, batchsize, snr_vals[i], phase)
                    q = data.build_prior_sampler(opt, batchsize)
            else:
                p = data.build_perturbed_data_sampler(opt, batchsize, snr_vals[i], phase)
                q = data.build_perturbed_data_sampler(opt, batchsize, snr_vals[i+1], phase)
                
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

        #validation end
        activate_policy(self.z_f)
        activate_policy(self.z_b)

        return {'forward_loss':forward_loss, 'backward_loss':backward_loss}

    def compute_interval_val_loss(self, opt, direction, dyn, ts):
        policy_opt, policy_impt = {
            'forward':  [self.z_f, self.z_b], # train forward, sample from backward
            'backward': [self.z_b, self.z_f], # train backward, sample from forward
        }.get(direction)

        batch_x = opt.samp_bs
        batch_t = ts.size(0)

        losses = []
        
        if hasattr(dyn.p, 'num_sample'):
            val_batches = int(opt.val_dataset_size * dyn.p.num_sample / dyn.p.batch_size)
        else:
            val_batches = opt.val_batches
            
        for _ in range(val_batches): #number of batches in the validation dataset.
            xs, zs_impt, ts_ = self.sample_train_data(opt, policy_impt, dyn, ts)
            xs.requires_grad_(True)
            xs = util.flatten_dim01(xs)
            zs_impt = util.flatten_dim01(zs_impt)
            ts_ = ts_.repeat(batch_x)
        
            assert xs.shape[0] == ts_.shape[0]
            assert zs_impt.shape[0] == ts_.shape[0]

            xs = xs.to(opt.device)
            zs_impt = zs_impt.to(opt.device)

            loss, zs = compute_sb_nll_alternate_train(
                    opt, dyn, ts_, xs, zs_impt, policy_opt, return_z=True
                )
            
            assert not torch.isnan(loss)
            
            #mem = float(torch.cuda.memory_allocated() / (1024 * 1024))
            #print("memory allocated:", mem, "MiB")
            
            losses.append(loss.item()) #important -> loss.item() 

            '''All loss tensors which are saved outside of the optimization cycle (i.e. outside the for g_iter in range(generator_iters) loop) 
            need to be detached from the graph. Otherwise, you are keeping all previous computation graphs in memory.'''
        
        #return the mean validation loss in that interval
        return np.mean(losses)

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

    def alternating_policy_update(self, opt, direction, interval_key, 
                                             policy_impt, policy_opt, outer_it,
                                             dyn, ts, stage_num, tr_steps=1):

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
            
            self.losses[direction][outer_it][stage_num]['train'][interval_key].append(loss.item())
            interval_steps = sum([len(self.losses[direction][outer_it][stage_num]['train'][interval_key]) for stage_num in self.losses[direction][outer_it].keys()])
            self.writer.add_scalar('outer_it_%d_train_%s_interval_%d' % (outer_it, direction, interval_key), loss, global_step=interval_steps)

            losses.append(loss.item())

        return losses

    def sb_outer_stage(self, opt, direction,
                               optimizer_f, optimizer_b, 
                               sched_f, sched_b, 
                               inter_pq_s, val_inter_pq_s, discretisation, 
                               tr_steps, outer_it, stage_num):

        policy_opt, policy_impt = {
            'forward':  [self.z_f, self.z_b], # train forward, sample from backward
            'backward': [self.z_b, self.z_f], # train backward, sample from forward
        }.get(direction)

        policy_impt = freeze_policy(policy_impt)
        policy_opt = activate_policy(policy_opt)

        start_inner_it = self.starting_inner_it[direction]

        #I need to modify the functionality of MultistageEarlyStoppingCallback
        #we need to replace train with val once we sort out the validation calculation.
        early_stopper = MultistageEarlyStoppingCallback(patience=opt.stopping_patience, loss_values=self.losses[direction][outer_it][stage_num]['train']) 
        
        #we should stop the training when we have convergence of all the intervals 
        #or we reach the max number of allowed iterations in one training episode
        for inner_it in tqdm(range(start_inner_it, opt.num_inner_iterations+1)):
            stop = early_stopper()
            if stop or inner_it == opt.num_inner_iterations:
                if stage_num == opt.num_stage: #save only in the last stage (at the end of each outer iteration)
                    #save the checkpoint before moving to the next outer iteration or finishing training.
                    keys = ['z_f','optimizer_f','ema_f','z_b','optimizer_b','ema_b']
                    util.multi_SBP_save_checkpoint(opt, self, keys, outer_it, stage_num, inner_it)
                    util.save_logs(opt, self, outer_it, stage_num, inner_it)

                #print samples at the end of every stage
                with torch.no_grad():
                    sample = self.multi_sb_generate_sample(opt, inter_pq_s, discretisation)
                    sample = sample.to('cpu')
                    #gt_sample = inter_pq_s[0][0].sample()
                
                img_dataset = util.is_image_dataset(opt)
                if img_dataset:
                    sample_imgs =  sample.cpu()
                    grid_images = torchvision.utils.make_grid(sample_imgs, nrow=int(np.sqrt(sample_imgs.size(0))), normalize=True, scale_each=True)
                    self.writer.add_image('outer_it:%d - stage:%d - direction:%s - inner_it:%d' % (outer_it, stage_num, direction, inner_it), grid_images)
                else:
                    problem_name = opt.problem_name
                    p, q = inter_pq_s[0]
                    dyn = sde.build(opt, p, q)
                    img = self.tensorboard_scatter_and_quiver_plot(opt, p, dyn, sample)
                    self.writer.add_image('outer_it:%d - stage:%d - direction:%s - inner_it:%d' % (outer_it, stage_num, direction, inner_it), img)

                break

            status = early_stopper.get_stages_status()
            probs = self.convergence_status_to_probs(status, opt.reweighting_factor)
            intervals = list(probs.keys())
            weights = [probs[key] for key in intervals]
            interval_key = random.choices(intervals, weights=weights, k=1)[0]
            p, q = inter_pq_s[interval_key-1]
            interval_dyn = sde.build(opt, p, q)
            ts = torch.linspace(p.time, q.time, discretisation)
            new_dt = ts[1]-ts[0]
            interval_dyn.dt = new_dt
            ts = ts.to(opt.device)

            losses = self.alternating_policy_update(opt, direction, interval_key, 
                                                    policy_impt, policy_opt, outer_it,
                                                    interval_dyn, ts, stage_num, tr_steps=tr_steps)
            avg_loss = np.mean(losses)
            early_stopper.add_stage_value(avg_loss, interval_key)

            self.global_step += 1
            self.logs['resume_info'][direction]['starting_inner_it'] = inner_it
            self.logs['resume_info'][direction]['global_step'] = self.global_step

            
            if self.global_step % opt.inner_it_save_freq == 0 and self.global_step !=0:
                keys = ['z_f','optimizer_f','ema_f','z_b','optimizer_b','ema_b']
                util.multi_SBP_save_checkpoint(opt, self, keys, outer_it, stage_num, inner_it)
                util.save_logs(opt, self, outer_it, stage_num, inner_it)
            
            '''
            if self.global_step % opt.sampling_freq == 0:
                with torch.no_grad():
                    sample = self.multi_sb_generate_sample(opt, inter_pq_s, discretisation)
                    sample = sample.to('cpu')
                    #gt_sample = inter_pq_s[0][0].sample()
                
                img_dataset = util.is_image_dataset(opt)
                if img_dataset:
                    sample_imgs =  sample.cpu()
                    grid_images = torchvision.utils.make_grid(sample_imgs, nrow=int(np.sqrt(sample_imgs.size(0))), normalize=True, scale_each=True)
                    self.writer.add_image('outer_it:%d - stage:%d - direction:%s - inner_it:%d' % (outer_it, stage_num, direction, inner_it), grid_images)
                else:
                    problem_name = opt.problem_name
                    p, q = inter_pq_s[0]
                    dyn = sde.build(opt, p, q)
                    img = self.tensorboard_scatter_and_quiver_plot(opt, p, dyn, sample)
                    self.writer.add_image('outer_it:%d - stage:%d - direction:%s - inner_it:%d' % (outer_it, stage_num, direction, inner_it), img)
            '''

            # We need to add the validation here. But let's skip it for the time being. 
            # It's the only thing missing. 
            # We then need to replace the train loss with the val loss in MultistageEarlyStoppingCallback

        #reset after the end of the every stage.
        self.starting_inner_it[direction] = 1
        self.logs['resume_info'][direction]['starting_inner_it'] = 1

    def sb_outer_alternating_iteration(self, opt,
                                            optimizer_f, optimizer_b, 
                                            sched_f, sched_b, 
                                            inter_pq_s, val_inter_pq_s, discretisation, 
                                            tr_steps, outer_it):

        start_inner_it = self.starting_inner_it
        early_stopper = MultistageEarlyStoppingCallback(patience=opt.stopping_patience, loss_values=self.losses['outer_it_%d' % outer_it]['val'])
        for inner_it in tqdm(range(start_inner_it, opt.num_inner_iterations+1)):
            stop = early_stopper()
            if stop or inner_it == opt.num_inner_iterations:
                #save the checkpoint before moving to the next outer iteration or finishing training.
                keys = ['z_f','optimizer_f','ema_f','z_b','optimizer_b','ema_b']
                util.multi_SBP_save_checkpoint(opt, self, keys, outer_it, inner_it)
                break
            
            status = early_stopper.get_stages_status()
            probs = self.convergence_status_to_probs(status, opt.reweighting_factor)
            intervals = list(probs.keys())
                
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
            self.losses['outer_it_%d' % outer_it]['train']['forward'][str(interval_key+1)].append(f_loss)

            losses = self.alternating_policy_update(opt, 'backward', interval_dyn, ts, tr_steps=tr_steps)
            b_loss = torch.mean(torch.tensor(losses)).item()
            self.losses['outer_it_%d' % outer_it]['train']['backward'][str(interval_key+1)].append(b_loss)

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

            self.writer.add_scalar('outer_it_%d_train_forward_loss_%d' % (outer_it, (interval_key+1)), f_loss, global_step=len(self.losses['outer_it_%d' % outer_it]['train']['forward'][str(interval_key+1)]))
            self.writer.add_scalar('outer_it_%d_train_backward_loss_%d' % (outer_it, (interval_key+1)), b_loss, global_step=len(self.losses['outer_it_%d' % outer_it]['train']['backward'][str(interval_key+1)]))

            self.z_f.starting_inner_it = torch.tensor(inner_it)
            self.starting_inner_it = inner_it
            self.z_f.global_step = torch.tensor(self.global_step)
            self.z_f.register_buffer('outer_it_%d_train_forward_loss_%d' % (outer_it, (interval_key+1)), torch.tensor(self.losses['outer_it_%d' % outer_it]['train']['forward'][str(interval_key+1)]))
            self.z_f.register_buffer('outer_it_%d_train_backward_loss_%d' % (outer_it, (interval_key+1)), torch.tensor(self.losses['outer_it_%d' % outer_it]['train']['backward'][str(interval_key+1)]))
            
            if inner_it % opt.val_freq == 0:
                #print('Validation starts...')
                val_loss = self.compute_val_loss(opt, val_inter_pq_s, discretisation)
                val_forward_loss = val_loss['forward_loss']
                val_backward_loss = val_loss['backward_loss']
                
                for interval_key in val_inter_pq_s.keys():
                    str_key = str(interval_key+1)
                    self.writer.add_scalar('outer_it_%d_val_forward_loss_%s' % (outer_it, str_key), val_forward_loss[str_key], global_step=len(self.losses['outer_it_%d' % outer_it]['val']['forward'][str_key]))
                    self.writer.add_scalar('outer_it_%d_val_backward_loss_%s' % (outer_it, str_key), val_backward_loss[str_key], global_step=len(self.losses['outer_it_%d' % outer_it]['val']['backward'][str_key]))
                    self.z_f.register_buffer('outer_it_%d_val_forward_loss_%s' % (outer_it, str_key), torch.tensor(self.losses['outer_it_%d' % outer_it]['val']['forward'][str_key]))
                    self.z_f.register_buffer('outer_it_%d_val_backward_loss_%s' % (outer_it, str_key), torch.tensor(self.losses['outer_it_%d' % outer_it]['val']['backward'][str_key]))
                    early_stopper.add_stage_value(val_forward_loss[str_key], 'forward', str_key)
                    early_stopper.add_stage_value(val_backward_loss[str_key], 'backward', str_key)
        
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
        
        tr_steps = opt.policy_updates
        for outer_it in range(self.starting_outer_it, outer_iterations+1):
            num_intervals = self.max_num_intervals//2**(outer_it-1)
            
            inter_pq_s = self.setup_intermediate_distributions(opt, self.level_log_SNR_max, self.level_log_SNR_min, 
                                                                    self.level_min_time, self.level_max_time, num_intervals,
                                                                    discretisation_policy=opt.discretisation_policy, outer_it=outer_it, phase='train')
            val_inter_pq_s = self.setup_intermediate_distributions(opt, self.level_log_SNR_max, self.level_log_SNR_min, 
                                                                    self.level_min_time, self.level_max_time, num_intervals, 
                                                                    discretisation_policy=opt.discretisation_policy, outer_it=outer_it, phase='val')
            
            new_discretisation = self.compute_discretisation(opt, outer_it)
            
            starting_stage = self.starting_stage if outer_it == self.starting_outer_it else 1

            #initialise the losses for the next outer iteration
            #self.starting_outer_it has been initialised in the init method
            if outer_it >= self.starting_outer_it+1:
                for direction in ['forward', 'backward']:
                    self.losses[direction][outer_it] = {}
                    self.losses[direction][outer_it][starting_stage]={}
                    for phase in ['train', 'val']:
                        self.losses[direction][outer_it][starting_stage][phase] = {}
                        for i in range(1, num_intervals+1):
                            self.losses[direction][outer_it][starting_stage][phase][i] = []

            for stage_num in range(starting_stage, opt.num_stage+1):
                if self.skip_backward and outer_it == self.starting_outer_it and stage_num == starting_stage:
                    self.sb_outer_stage(opt, 'forward',
                                      optimizer_f, optimizer_b, sched_f, sched_b, 
                                      inter_pq_s, val_inter_pq_s, new_discretisation, 
                                      tr_steps, outer_it, stage_num)
                    self.logs['resume_info']['forward']['starting_stage'] += 1

                else:
                    if stage_num > starting_stage:
                        for direction in ['forward', 'backward']:
                            self.losses[direction][outer_it][stage_num] = {}
                            for phase in ['train', 'val']:
                                self.losses[direction][outer_it][stage_num][phase]={}
                                for i in range(1, num_intervals+1):
                                    self.losses[direction][outer_it][stage_num][phase][i] = []

                    self.sb_outer_stage(opt, 'backward',
                                        optimizer_f, optimizer_b, sched_f, sched_b, 
                                        inter_pq_s, val_inter_pq_s, new_discretisation, 
                                        tr_steps, outer_it, stage_num)
                    self.logs['resume_info']['backward']['starting_stage'] += 1
                    
                    self.sb_outer_stage(opt, 'forward',
                                        optimizer_f, optimizer_b, sched_f, sched_b, 
                                        inter_pq_s, val_inter_pq_s, new_discretisation, 
                                        tr_steps, outer_it, stage_num)
                    self.logs['resume_info']['forward']['starting_stage'] += 1

                    
            
            self.logs['resume_info']['forward']['starting_outer_it']+=1
            self.logs['resume_info']['backward']['starting_outer_it']+=1
    
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
        num_intervals = self.max_num_intervals // 2**(outer_it-1)
        inter_pq_s = self.setup_intermediate_distributions(opt, self.level_log_SNR_max, self.level_log_SNR_min, 
                                                                self.level_min_time, self.level_max_time, num_intervals)

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

    #this method needs to be modified
    @torch.no_grad()
    def sample(self, opt, discretisation, x=None, save_traj=True, stochastic=True):
        #1.) detect number of SBP stages
        outer_it = self.z_f.starting_outer_it.item()
        num_intervals = self.max_num_intervals // 2**(outer_it-1)
        inter_pq_s = self.setup_intermediate_distributions(opt, self.level_log_SNR_max, self.level_log_SNR_min, 
                                                                self.level_min_time, self.level_max_time, num_intervals)

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
                if x is None:
                    x = q.sample().to(opt.device)

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
            
        return {'trajectory': xs, 'sample': x}
    
    @torch.no_grad()
    def multi_sb_generate_sample(self, opt, inter_pq_s, discretisation, initial_sample=None):
        sorted_keys = sorted(list(inter_pq_s.keys()), reverse=True)
        for i, key in enumerate(sorted_keys):
            p, q = inter_pq_s[key]
            interval_dyn = sde.build(opt, p, q)
            ts = torch.linspace(p.time, q.time, discretisation)
            new_dt = ts[1]-ts[0]
            interval_dyn.dt = new_dt
            ts = ts.to(opt.device)
            
            if i==0 and initial_sample is None:
                if not self.last_level:
                    max_level = max(inter_pq_s.keys())
                    initial_sample = inter_pq_s[max_level][1].sample().to(opt.device)
            
            _, _, initial_sample = interval_dyn.sample_traj(ts, self.z_b,
                                                            save_traj=False,
                                                            initial_sample=initial_sample)

        return initial_sample

    def log_tb(self, step, val, name, tag):
        self.writer.add_scalar(os.path.join(tag,name), val, global_step=step)
