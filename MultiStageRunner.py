
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
from loss import compute_sb_nll_alternate_train, compute_sb_nll_joint_train, compute_sb_nll_joint_increment
import data
import util

from ipdb import set_trace as debug
import random
from tqdm import tqdm
import copy
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

def initialise_logs(train_method, num_intervals, reduction_levels):
    logs = {}
    logs['num_intervals'] = num_intervals
    logs['reduction_levels'] = reduction_levels
    logs['resume_info'] = {}
    logs['loss'] = {}

    outer_it = 1
    
    if train_method == 'alternate':
        stage = 1
        logs['loss']['val_increment_loss'] = {}
        logs['loss']['val_increment_loss'][outer_it] = []

        for direction in ['forward', 'backward']:
            logs['resume_info'][direction] = {}
            logs['resume_info'][direction]['starting_outer_it'] = 1
            logs['resume_info'][direction]['starting_stage'] = 1
            logs['resume_info'][direction]['starting_inner_it'] = 1
            logs['resume_info'][direction]['global_step'] = 1
            logs['loss'][direction] = {}
            logs['loss'][direction][outer_it] = {}
            logs['loss'][direction][outer_it][stage]={}
            for phase in ['train', 'val']:
                logs['loss'][direction][outer_it][stage][phase] = {}
                for i in range(1, num_intervals+1):
                    logs['loss'][direction][outer_it][stage][phase][i] = []

    elif train_method == 'joint':
        logs['resume_info']['starting_outer_it'] = outer_it
        logs['resume_info']['starting_inner_it'] = 1
        logs['resume_info']['global_step'] = 1 
        logs['loss'][outer_it]={}
        logs['loss'][outer_it]['val']=[]
        logs['loss'][outer_it]['train']={}
        for i in range(1, num_intervals+1):
            logs['loss'][outer_it]['train'][i]=[]

    return logs

#create a new class that inherits the class of all models in reduced level j and can perform sampling.
class MultistageCombiner():
    def __init__(self, opt):
        self.multistage_model = {}
        self.opts = {}
        for i in range(1, opt.reduction_levels+1):
            opt.level_id = i
            opt.load = opt.reduced_models_load[i-1]
            multistage_phase_path = os.path.join(opt.experiment_path, 'reduction_%d' % opt.reduction_levels, '%d' % opt.level_id)
            opt.ckpt_path = os.path.join(multistage_phase_path, 'checkpoints')
            opt.logs_path = os.path.join(multistage_phase_path, 'logs')

            self.opts[i] = opt

            print(self.opts[i].level_id)
            print(self.opts[i].load)
            self.multistage_model[i] = MultiStageRunner(self.opts[i])
        
        log_dir = os.path.join(opt.experiment_path,  'reduction_%d' % opt.reduction_levels, 'samples')
        self.writer = SummaryWriter(log_dir=log_dir)
    
    def sample(self, save_traj=True, stochastic=False, target_level=1):
        levels = sorted(list(self.opts.keys()), reverse=True)
        x = None
        for level in levels:
            opt = self.opts[level]
            out = self.multistage_model[level].sample(opt, opt.base_discretisation, x, save_traj, stochastic)
            x = out['sample']

            if level == target_level:
                break
        
        return x
    
    def compute_fid(self, generate_dataset=True):
        base_opt = self.opts[1]
        FID_path = util.get_FID_npz_path(base_opt)

        if generate_dataset:
            self.generate_FID_dataset(target_level=base_opt.target_level)
            
        fid_val = util.get_fid(FID_path, base_opt.eval_target_level_path)
        print('FID: ', fid_val) 

    def generate_FID_dataset(self, num_samples=50000, stochastic=True, target_level=1):
        #basic implementation (no tricks yet)
        base_opt = self.opts[1]
        batchsize = base_opt.samp_bs

        passes = num_samples // batchsize + 1
        extra_part = num_samples - (num_samples // batchsize) * batchsize

        counter = 1
        for i in tqdm(range(1, passes+1)):
            x = self.sample(save_traj=False, stochastic=stochastic, target_level=target_level)
            x = util.norm_data(base_opt, x)
            
            if i == passes:
                x = x[:extra_part]
            
            for i in range(x.shape[0]):
                fn = os.path.join(base_opt.eval_target_level_path, 'img{}.jpg'.format(counter))
                torchvision.utils.save_image(x[i,...], fn)
                counter += 1

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

        self.load = True if opt.load else False
        if opt.load:
            checkpoint_path = os.path.join(opt.ckpt_path, opt.load + '.npz')
            util.restore_checkpoint(opt, self, checkpoint_path)

            if self.reduction_levels < self.z_f.reduction_levels.item():
                #in this case we enter a new reduction cycle
                #we need to initialize the logs
                print('New option reduction levels less than loaded reduction levels.')
                print('We are entering a new reduction cycle.')
                self.logs = initialise_logs(opt.train_method, self.max_num_intervals, self.reduction_levels)
            else:
                print('New option reduction levels same as the loaded reduction levels')
                print('We are either resuming training or using the loaded model for sampling.')
                logs_path = os.path.join(opt.logs_path, opt.load + '.pkl')
                self.logs = util.restore_logs(logs_path)
                stop_info = opt.load.split('_')

                #In this case we just finished the previous outer iteration 
                #so we should inform the sampler that we self.starting_outer_it for sampling should be reduced by 1. 
                #The resuming starts from the next outer iteration which is saved as the default starting outer it value.
                if stop_info[1]=='1' and stop_info[2]=='1':
                    print('We are reducing the starting outer it by 1.')
                    self.reduce_outer_it_in_sampling = 1
                else:
                    print('We keep the same outer it as the one loaded from the model.')
                    self.reduce_outer_it_in_sampling = 0

        else:
            self.logs = initialise_logs(opt.train_method, self.max_num_intervals, self.reduction_levels)
        
        #every should be read/written from/to self.logs from now on.
        if opt.train_method == 'alternate':
            self.starting_outer_it = self.logs['resume_info']['forward']['starting_outer_it']
            self.starting_stage = self.logs['resume_info']['forward']['starting_stage']
            self.skip_backward = True if self.logs['resume_info']['backward']['starting_stage'] > self.logs['resume_info']['forward']['starting_stage'] else False
            self.global_step = self.logs['resume_info']['forward']['global_step']
            self.starting_inner_it = {'forward': self.logs['resume_info']['forward']['starting_inner_it'],
                                  'backward': self.logs['resume_info']['backward']['starting_inner_it']}
        elif opt.train_method == 'joint':
            self.starting_outer_it = self.logs['resume_info']['starting_outer_it']
            self.starting_inner_it = self.logs['resume_info']['starting_inner_it']
            self.global_step = self.logs['resume_info']['global_step']

        self.losses = self.logs['loss']
        print(self.losses.keys())

        if opt.log_tb: # tensorboard related things
            self.it_f = 0
            self.it_b = 0

            log_dir = os.path.join(opt.experiment_path,  'reduction_%d' % opt.reduction_levels, '%d' % opt.level_id, 'tensorboard_logs')
            self.log_dir = log_dir
            self.writer = SummaryWriter(log_dir=log_dir)

    #done
    def setup_equidistant_distributions(self, opt, log_SNR_max, log_SNR_min, min_time, max_time, num_intervals, \
                                            discretisation_policy='constant', outer_it=1, phase='train'):
        if discretisation_policy == 'constant':
            batchsize = opt.samp_bs
        elif discretisation_policy == 'double':
            batchsize = opt.samp_bs // 2**(outer_it-1)

        snr0=np.exp(log_SNR_max)
        a1 = np.sqrt(snr0/(1+snr0))
        s1 = np.sqrt(1/(1+snr0))

        target_sigma = opt.prior_std #we need to set this carefully. Default value: 0.5
        target_logsnr = log_SNR_min
        target_alpha = np.sqrt(target_sigma**2*np.exp(target_logsnr))

        #derive transition parameters
        targetN = num_intervals + 1
        aT=(target_alpha/a1)**(1/(targetN-1))
        sT=np.sqrt((target_sigma**2-aT**(targetN-1)*s1**2)*(1-aT**2)/(1-(aT**2)**(targetN-1)))
        s_infty = sT/np.sqrt(1-aT**2) #this must be equal to prior_std

        def get_alpha_fn(a1, aT):
            def alpha_fn(k):
                aK = aT**(k-1)*a1
                return aK
            return alpha_fn
        
        def get_sigma_fn(sT, aT, s1):
            def sigma_fn(k):
                sK = np.sqrt(sT**2*(1-(aT**2)**(k-1))/(1-aT**2)+aT**(k-1)*s1**2)
                return sK
            return sigma_fn
        
        alpha_fn = get_alpha_fn(a1, aT)
        sigma_fn = get_sigma_fn(sT, aT, s1)

        alphas = [alpha_fn(k) for k in np.arange(1, num_intervals+1)]
        sigmas = [sigma_fn(k) for k in np.arange(1, num_intervals+1)]
        times = torch.linspace(min_time, max_time, num_intervals+1)

        intermediate_distributions = {}
        for i in range(num_intervals):
            if self.last_level:
                if i < num_intervals - 1:
                    p = data.build_equidistant_perturbed_data_sampler(opt, batchsize, alphas[i], sigmas[i], phase)
                    q = data.build_equidistant_perturbed_data_sampler(opt, batchsize, alphas[i+1], sigmas[i+1], phase)
                elif i == num_intervals - 1:
                    p = data.build_equidistant_perturbed_data_sampler(opt, batchsize, alphas[i], sigmas[i], phase)
                    q = data.build_prior_sampler(opt, batchsize)
            else:
                p = data.build_equidistant_perturbed_data_sampler(opt, batchsize, alphas[i], sigmas[i], phase)
                q = data.build_equidistant_perturbed_data_sampler(opt, batchsize, alphas[i+1], sigmas[i+1], phase)
                
            p.time = times[i]
            p.aT = aT
            p.sT = sT
            q.time = times[i+1]
            intermediate_distributions[i] = [p, q]

        return intermediate_distributions

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
        #assert xs.shape[0] == opt.samp_bs
        #assert xs.shape[1] == len(ts)
        #assert xs.shape == zs.shape
        gc.collect()
        return xs, zs, ts
    
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

    #This is what we should use to assess convergence in the alternating training procedure
    def compute_level_contribution_to_ll(self, opt, inter_pq_s, discretisation): 
        num_batches = 20
        average_total_increment = 0.
        for j in range(num_batches):
            total_increment = 0.
            sorted_keys = sorted(list(inter_pq_s.keys()))
            for i, key in enumerate(sorted_keys):
                p, q = inter_pq_s[key]
                dyn = sde.build(opt, p, q)
                ts = torch.linspace(p.time, q.time, discretisation)
                new_dt = ts[1]-ts[0]
                dyn.dt = new_dt
                ts = ts.to(opt.device)

                with torch.no_grad():
                    xs_f, zs_f, x_term_f, orig_x = dyn.sample_traj(ts, self.z_f, save_traj=True, return_original=True)

                batch_x = xs_f.size(0)

                xs_f.requires_grad_(True)
                zs_f.requires_grad_(True)
                xs_f=util.flatten_dim01(xs_f)
                zs_f=util.flatten_dim01(zs_f)
                ts_=ts.repeat(batch_x)

                if not self.last_level:
                    x_term_f = None
                else:
                    orig_x = None
                    if key == max(sorted_keys):
                        x_term_f.requires_grad_(True)
                    else:
                        x_term_f = None

                '''
                if key == max(sorted_keys):
                    x_term_f.requires_grad_(True)
                    if self.last_level:
                        orig_x = None
                else:
                    x_term_f = None
                '''
                
                interval_increment = compute_sb_nll_joint_increment(opt, batch_x, dyn, ts_, xs_f, zs_f, self.z_b, x_term_f, orig_x)
                total_increment += interval_increment.item()

            average_total_increment += total_increment

        average_total_increment /= num_batches
        return average_total_increment

    def sb_joint_outer_iteration(self, opt, optimizer_f, optimizer_b, sched_f, sched_b, 
                                 inter_pq_s, val_inter_pq_s, discretisation, 
                                 tr_steps, outer_it):
        
        sorted_keys = sorted(list(inter_pq_s.keys()))
        start_inner_it = copy.copy(self.starting_inner_it)
        early_stopper = EarlyStoppingCallback(patience=opt.stopping_patience, loss_values=self.losses[outer_it]['val'])
        for inner_it in tqdm(range(start_inner_it, opt.num_inner_iterations+1)):
            stop = early_stopper()
            if stop or inner_it == opt.num_inner_iterations:
                #save the checkpoint before moving to the next outer iteration or finishing training.
                keys = ['z_f','optimizer_f','ema_f','z_b','optimizer_b','ema_b']
                save_name = '%d_%d' % (outer_it, inner_it)
                util.multi_SBP_save_checkpoint(opt, self, keys, save_name)
                util.save_logs(opt, self, save_name)
                break

            interval_key = random.choice(sorted_keys)
            p, q = inter_pq_s[interval_key]

            interval_dyn = sde.build(opt, p, q)
            ts = torch.linspace(p.time, q.time, discretisation)
            new_dt = ts[1]-ts[0]
            interval_dyn.dt = new_dt
            ts = ts.to(opt.device)

            optimizer_f.zero_grad()
            optimizer_b.zero_grad()

            xs_f, zs_f, x_term_f, orig_x = interval_dyn.sample_traj(ts, self.z_f, save_traj=True, return_original=True)
            
            #xs_f.requires_grad_(True)
            #zs_f.requires_grad_(True)

            batch_x = xs_f.size(0)
            xs_f = util.flatten_dim01(xs_f)
            zs_f = util.flatten_dim01(zs_f)
            ts_ = ts.repeat(batch_x)

            if interval_key == max(sorted_keys) and self.last_level:
                orig_x = None

            loss = compute_sb_nll_joint_increment(opt, batch_x, interval_dyn, ts_, xs_f, zs_f, self.z_b, x_term_f, orig_x)
            loss.backward()

            optimizer_f.step()
            optimizer_b.step()

            if sched_f is not None: sched_f.step()
            if sched_b is not None: sched_b.step()

            loss_val = loss.item()
            self.losses[outer_it]['train'][interval_key+1].append(loss_val)
            train_steps = len(self.losses[outer_it]['train'][interval_key+1])
            self.writer.add_scalar('outer_it_%d_train_loss_%d' % (outer_it, (interval_key+1)), loss_val, global_step=train_steps)

            if self.global_step % opt.inner_it_save_freq == 0:
                keys = ['z_f','optimizer_f','ema_f','z_b','optimizer_b','ema_b']
                save_name = '%d_%d' % (outer_it, inner_it)
                util.multi_SBP_save_checkpoint(opt, self, keys, save_name)
                util.save_logs(opt, self, save_name)

            if self.global_step % opt.sampling_freq == 0:
                print('----- sampling -----')
                with torch.no_grad():
                    sample = self.multi_sb_generate_sample(opt, inter_pq_s, discretisation)
                    sample = sample.to('cpu')
                    #gt_sample = inter_pq_s[0][0].sample()
                
                img_dataset = util.is_image_dataset(opt)
                if img_dataset:
                    sample_imgs =  sample.cpu()
                    grid_images = torchvision.utils.make_grid(sample_imgs, normalize=True, scale_each=True)
                    self.writer.add_image('outer_it:%d - inner_it:%d' % (outer_it, inner_it), grid_images, self.global_step)
                else:
                    problem_name = opt.problem_name
                        
                    p, q = inter_pq_s[0]
                    dyn = sde.build(opt, p, q)
                    img = self.tensorboard_scatter_and_quiver_plot(opt, p, dyn, sample)
                    self.writer.add_image('samples and ODE vector field, outer_it:%d - inner_it:%d' % (outer_it, inner_it), img)
            
            self.global_step += 1
            self.starting_inner_it = inner_it

            self.logs['resume_info']['starting_inner_it'] = inner_it
            self.logs['resume_info']['global_step'] = self.global_step
            
            if inner_it % opt.val_freq == 0:
                #print('Validation starts...')
                #freeze_policy(self.z_f)
                #freeze_policy(self.z_b)
                
                val_loss = self.compute_level_contribution_to_ll(opt, val_inter_pq_s, discretisation)
                self.losses[outer_it]['val'].append(val_loss)
                val_steps = len(self.losses[outer_it]['val'])
                self.writer.add_scalar('outer_it_%d_val_loss' % outer_it, val_loss, global_step=val_steps)

                #activate_policy(self.z_f)
                #activate_policy(self.z_b)

        
        #reset after the end of the outer iteration.
        self.starting_inner_it = 1
        self.logs['resume_info']['starting_inner_it'] = self.starting_inner_it

    def sb_joint_train(self, opt):
        policy_f, policy_b = self.z_f, self.z_b
        policy_f = activate_policy(policy_f)
        policy_b = activate_policy(policy_b)
        optimizer_f, _, sched_f = self.get_optimizer_ema_sched(policy_f)
        optimizer_b, _, sched_b = self.get_optimizer_ema_sched(policy_b)
        outer_iterations = self.num_outer_iterations 
        tr_steps = opt.policy_updates

        starting_outer_it = copy.copy(self.starting_outer_it)
        for outer_it in range(starting_outer_it, outer_iterations+1):
            num_intervals = self.max_num_intervals//2**(outer_it-1)
            
            inter_pq_s = self.setup_intermediate_distributions(opt, self.level_log_SNR_max, self.level_log_SNR_min, 
                                                                    self.level_min_time, self.level_max_time, num_intervals,
                                                                    discretisation_policy=opt.discretisation_policy, outer_it=outer_it, phase='train')
            val_inter_pq_s = self.setup_intermediate_distributions(opt, self.level_log_SNR_max, self.level_log_SNR_min, 
                                                                    self.level_min_time, self.level_max_time, num_intervals, 
                                                                    discretisation_policy=opt.discretisation_policy, outer_it=outer_it, phase='val')
            
            new_discretisation = self.compute_discretisation(opt, outer_it)
            
            if outer_it not in self.losses.keys():
                self.logs['loss'][outer_it]={}
                self.logs['loss'][outer_it]['val']=[]
                self.logs['loss'][outer_it]['train']={}
                for i in range(1, num_intervals+1):
                    self.logs['loss'][outer_it]['train'][i]=[]

            self.sb_joint_outer_iteration(opt, optimizer_f, 
                                    optimizer_b, sched_f, sched_b, 
                                    inter_pq_s, val_inter_pq_s, new_discretisation, 
                                    tr_steps, outer_it)
            
            self.logs['resume_info']['starting_outer_it']+=1
            self.starting_outer_it += 1

    def alternating_policy_update(self, opt, direction, interval_key, 
                                             policy_impt, policy_opt, outer_it,
                                             dyn, ts, stage_num, tr_steps=1):

        optimizer, ema, sched = self.get_optimizer_ema_sched(policy_opt)

        #batch_x = opt.samp_bs
        #batch_t = ts.size(0)
        losses = []
        for it in range(tr_steps):
            optimizer.zero_grad()

            xs, zs_impt, ts_ = self.sample_train_data(opt, policy_impt, dyn, ts)
            batch_x = xs.size(0)

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

    def modified_score_matching(self, opt, direction, interval_key, 
                                                    policy_impt, policy_opt, outer_it,
                                                    interval_dyn, tr_steps, stage_num):
        #batch_size = interval_dyn.p.batch_size
        t0 = interval_dyn.p.time + torch.tensor(1e-6) #add this for stability
        t1 = interval_dyn.q.time
        optimizer, ema, sched = self.get_optimizer_ema_sched(policy_opt)
        losses = []
        for it in range(tr_steps):
            optimizer.zero_grad()

            x0, x1 = interval_dyn.get_paired_samples(direction)
            t = torch.rand(x0.size(0)).to(opt.device) * (t1 - t0) + t0 
            #print(x0.size())
            #print(x1.size())
            xt = interval_dyn.get_sample_from_posterior_given_pair(t, x0, x1)
            sigma_t = torch.sqrt(interval_dyn.forward_variance_accumulation(t))

            epsilon = policy_opt(xt, t)
            loss = epsilon - (xt - x0)/sigma_t[(...,) + (None,) * len(x0.shape[1:])]
            loss = torch.mean(torch.square(loss))

            loss.backward()
            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm(policy_opt.parameters(), opt.grad_clip)
            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            self.losses[direction][outer_it][stage_num]['train'][interval_key].append(loss.item())
            interval_steps = sum([len(self.losses[direction][outer_it][stage_num]['train'][interval_key]) for stage_num in self.losses[direction][outer_it].keys()])
            self.writer.add_scalar('outer_it_%d_train_%s_interval_%d' % (outer_it, direction, interval_key), loss, global_step=interval_steps)

            #print(loss.item())
            losses.append(loss.item())
        
        return losses

    def first_outer_it_with_score_matching(self, opt, direction,
                               inter_pq_s, val_inter_pq_s, discretisation, 
                               tr_steps):
        
        outer_it = 1
        stage_num = 1
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
                break
            
            #if inner_it == 2:
            #    self.encode_and_visualise_trajectories(opt, inter_pq_s)
            #    break

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

            #I need to create a new update policy that performs score matching according to equation 12
            #while the sample is drawn from the posterior distribution conditioned on the endpoints X0,X1 (equation 11)
            #I need samples from the transition kernel that maps p to q.
            losses = self.modified_score_matching(opt, direction, interval_key, 
                                                    policy_impt, policy_opt, outer_it,
                                                    interval_dyn, tr_steps, stage_num)
            avg_loss = np.mean(losses)
            early_stopper.add_stage_value(avg_loss, interval_key)

            self.global_step += 1
            self.logs['resume_info'][direction]['starting_inner_it'] = inner_it
            self.logs['resume_info'][direction]['global_step'] = self.global_step

            if self.global_step % opt.sampling_freq == 0 and self.global_step !=0:
                #print samples for every saved checkpoint
                with torch.no_grad():
                    #sample = self.multi_sb_generate_sample(opt, inter_pq_s, discretisation)
                    sample = self.ddpm_sample(opt, inter_pq_s, discretisation)
                    sample = sample.to('cpu')
                    #gt_sample = inter_pq_s[0][0].sample()
                
                img_dataset = util.is_image_dataset(opt)
                if img_dataset:
                    sample_imgs =  sample.cpu()
                    grid_images = torchvision.utils.make_grid(sample_imgs, nrow=int(np.sqrt(sample_imgs.size(0))), normalize=True, scale_each=True)
                    self.writer.add_image('outer_it:%d - stage:%d - direction:%s - inner_it:%d' % (outer_it, stage_num, direction, inner_it), grid_images)
                else:
                    problem_name = opt.problem_name
                    #p, q = inter_pq_s[0]
                    #dyn = sde.build(opt, p, q)
                    #img = self.tensorboard_scatter_and_quiver_plot(opt, p, dyn, sample)
                    img = self.tensorboard_scatter_plot(sample, problem_name, inner_it, outer_it)
                    self.writer.add_image('outer_it:%d - stage:%d - direction:%s - inner_it:%d' % (outer_it, stage_num, direction, inner_it), img)

            if self.global_step % opt.inner_it_save_freq == 0 and self.global_step !=0:
                keys = ['z_f','optimizer_f','ema_f','z_b','optimizer_b','ema_b']
                save_name = '%d_%d_%d' % (outer_it, stage_num, inner_it)
                util.multi_SBP_save_checkpoint(opt, self, keys, save_name)
                util.save_logs(opt, self, save_name)              
            

            # We need to add the validation here. But let's skip it for the time being. 
            # It's the only thing missing. 
            # We then need to replace the train loss with the val loss in MultistageEarlyStoppingCallback

        #reset after the end of the every stage.
        self.starting_inner_it[direction] = 1
        self.logs['resume_info'][direction]['starting_inner_it'] = 1

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

            if self.global_step % opt.sampling_freq == 0 and self.global_step !=0:
                #print samples for every saved checkpoint
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

            if self.global_step % opt.inner_it_save_freq == 0 and self.global_step !=0:
                keys = ['z_f','optimizer_f','ema_f','z_b','optimizer_b','ema_b']
                save_name = '%d_%d_%d' % (outer_it, stage_num, inner_it)
                util.multi_SBP_save_checkpoint(opt, self, keys, save_name)
                util.save_logs(opt, self, save_name)              
            

            # We need to add the validation here. But let's skip it for the time being. 
            # It's the only thing missing. 
            # We then need to replace the train loss with the val loss in MultistageEarlyStoppingCallback

        #reset after the end of the every stage.
        self.starting_inner_it[direction] = 1
        self.logs['resume_info'][direction]['starting_inner_it'] = 1

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
            
            '''
            inter_pq_s = self.setup_intermediate_distributions(opt, self.level_log_SNR_max, self.level_log_SNR_min, 
                                                                    self.level_min_time, self.level_max_time, num_intervals,
                                                                    discretisation_policy=opt.discretisation_policy, outer_it=outer_it, phase='train')
            val_inter_pq_s = self.setup_intermediate_distributions(opt, self.level_log_SNR_max, self.level_log_SNR_min, 
                                                                    self.level_min_time, self.level_max_time, num_intervals, 
                                                                    discretisation_policy=opt.discretisation_policy, outer_it=outer_it, phase='val')
            '''
            inter_pq_s = self.setup_equidistant_distributions(opt, self.level_log_SNR_max, self.level_log_SNR_min, 
                                                                    self.level_min_time, self.level_max_time, num_intervals,
                                                                    discretisation_policy=opt.discretisation_policy, outer_it=outer_it, phase='train')
            val_inter_pq_s = self.setup_equidistant_distributions(opt, self.level_log_SNR_max, self.level_log_SNR_min, 
                                                                    self.level_min_time, self.level_max_time, num_intervals, 
                                                                    discretisation_policy=opt.discretisation_policy, outer_it=outer_it, phase='val')


            new_discretisation = self.compute_discretisation(opt, outer_it)
            
            starting_stage = self.starting_stage if outer_it == self.starting_outer_it else 1

            #initialise the losses for the next outer iteration
            #self.starting_outer_it has been initialised in the init method

            initialisation_condition = outer_it not in self.losses['val_increment_loss'].keys()
            if initialisation_condition:
                self.losses['val_increment_loss'][outer_it] = []
                for direction in ['forward', 'backward']:
                    self.losses[direction][outer_it] = {}
                    self.losses[direction][outer_it][starting_stage]={}
                    for phase in ['train', 'val']:
                        self.losses[direction][outer_it][starting_stage][phase] = {}
                        for i in range(1, num_intervals+1):
                            self.losses[direction][outer_it][starting_stage][phase][i] = []
            
            stages_early_stopper = EarlyStoppingCallback(patience=opt.stage_patience, loss_values=self.losses['val_increment_loss'][outer_it])
            for stage_num in range(starting_stage, opt.num_stage+1):
                stop = stages_early_stopper()
                if stop or stage_num == opt.num_stage:
                    for direction in ['forward', 'backward']:
                        self.logs['resume_info'][direction]['starting_outer_it'] = outer_it+1
                        self.logs['resume_info'][direction]['starting_stage'] = 1
                        self.logs['resume_info'][direction]['starting_inner_it'] = 1

                    keys = ['z_f','optimizer_f','ema_f','z_b','optimizer_b','ema_b']
                    save_name = '%d_%d_%d' % (outer_it+1, 1, 1)
                    util.multi_SBP_save_checkpoint(opt, self, keys, save_name)
                    util.save_logs(opt, self, save_name)
                    break
                    
                if self.skip_backward and outer_it == self.starting_outer_it and stage_num == starting_stage:
                    self.sb_outer_stage(opt, 'forward',
                                      optimizer_f, optimizer_b, sched_f, sched_b, 
                                      inter_pq_s, val_inter_pq_s, new_discretisation, 
                                      tr_steps, outer_it, stage_num)
                    self.logs['resume_info']['forward']['starting_stage'] += 1

                    val_increment_loss = self.compute_level_contribution_to_ll(opt, val_inter_pq_s, new_discretisation)
                    self.losses['val_increment_loss'][outer_it].append(val_increment_loss)
                    step = len(self.losses['val_increment_loss'][outer_it])
                    self.writer.add_scalar('val_increment_loss_outer_it_%d' % outer_it, val_increment_loss, global_step=step)
                    stages_early_stopper.add_value(val_increment_loss)
                else:
                    if stage_num > starting_stage:
                        for direction in ['forward', 'backward']:
                            self.losses[direction][outer_it][stage_num] = {}
                            for phase in ['train', 'val']:
                                self.losses[direction][outer_it][stage_num][phase]={}
                                for i in range(1, num_intervals+1):
                                    self.losses[direction][outer_it][stage_num][phase][i] = []

                    self.first_outer_it_with_score_matching(opt, 'backward',
                               inter_pq_s, val_inter_pq_s, new_discretisation, tr_steps)

                    '''
                    self.sb_outer_stage(opt, 'backward',
                                        optimizer_f, optimizer_b, sched_f, sched_b, 
                                        inter_pq_s, val_inter_pq_s, new_discretisation, 
                                        tr_steps, outer_it, stage_num)
                    self.logs['resume_info']['backward']['starting_stage'] += 1

                    #compute the contribution to joint loglikelihood of the current level 
                    #after the end of training stage.
                    #If this stops improving we can move to the next outer iteration.
                    
                    if stage_num % opt.val_freq == 0:
                        val_increment_loss = self.compute_level_contribution_to_ll(opt, val_inter_pq_s, new_discretisation)
                        self.losses['val_increment_loss'][outer_it].append(val_increment_loss)
                        step = len(self.losses['val_increment_loss'][outer_it])
                        self.writer.add_scalar('val_increment_loss_outer_it_%d' % outer_it, val_increment_loss, global_step=step)
                        stages_early_stopper.add_value(val_increment_loss)
                    

                    self.sb_outer_stage(opt, 'forward',
                                        optimizer_f, optimizer_b, sched_f, sched_b, 
                                        inter_pq_s, val_inter_pq_s, new_discretisation, 
                                        tr_steps, outer_it, stage_num)
                    self.logs['resume_info']['forward']['starting_stage'] += 1

                    if stage_num % opt.val_freq == 0:
                        val_increment_loss = self.compute_level_contribution_to_ll(opt, val_inter_pq_s, new_discretisation)
                        self.losses['val_increment_loss'][outer_it].append(val_increment_loss)
                        step = len(self.losses['val_increment_loss'][outer_it])
                        self.writer.add_scalar('val_increment_loss_outer_it_%d' % outer_it, val_increment_loss, global_step=step)
                        stages_early_stopper.add_value(val_increment_loss)
                    
                    '''

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
            ts = torch.flip(ts,dims=[0]) #?? this line should probably be eliminated

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
        if hasattr(self, 'sampling_inter_pq_s'):
            inter_pq_s = self.sampling_inter_pq_s
        else:
            #1.) detect number of SBP stages
            outer_it = self.starting_outer_it - self.reduce_outer_it_in_sampling
            num_intervals = self.max_num_intervals // 2**(outer_it-1)
            inter_pq_s = self.setup_intermediate_distributions(opt, self.level_log_SNR_max, self.level_log_SNR_min, 
                                                                    self.level_min_time, self.level_max_time, num_intervals)
            self.sampling_inter_pq_s = inter_pq_s

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

                xs = torch.empty((x.size(0), (discretisation)*num_intervals+1, *x.shape[1:])) if save_traj else None

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
                    xs[:,i*(discretisation)+idx+1,::] = x.detach().cpu()
            
        return {'trajectory': xs, 'sample': x}
    
    @torch.no_grad()
    def encode_and_visualise_trajectories(self, opt, inter_pq_s):
        save_path = os.path.join(opt.experiment_path, 'testing')
        os.makedirs(save_path, exist_ok=True)

        sorted_keys = sorted(list(inter_pq_s.keys()))
        sample_evolution = []
        for i, key in tqdm(enumerate(sorted_keys)):
            p, q = inter_pq_s[key]
            interval_dyn = sde.build(opt, p, q)
            if i==0:
                x0 = p.sample()
                sample_evolution.append(x0)

            aT = p.aT
            sT = p.sT
            x0 = aT * x0 + sT * torch.randn_like(x0)
            sample_evolution.append(x0)
        
        traj = torch.stack(sample_evolution)
        print(traj.size())
        
        plt.figure()
        for i in range(traj.size(1)):
            color = (np.random.rand(), np.random.rand(), np.random.rand())
            plt.plot(traj[500:502,i,0], traj[500:502,i,1], color=color, alpha=0.3)

        #plt.scatter(x[:,0], x[:,1])
        plt.savefig(os.path.join(save_path, 'encoded_trajectory.png' ))

        #return sample_evolution


    @torch.no_grad()
    def ddpm_sample(self, opt, inter_pq_s, discretisation, return_evolution=False, starting_level=1):
        #self.z_b predicts x0 in this implementation.
        if return_evolution:
            evolution=[]

        sorted_keys = sorted(list(inter_pq_s.keys()), reverse=True)
        if starting_level > 1:
            sorted_keys = sorted_keys[starting_level:]

        for i, key in tqdm(enumerate(sorted_keys)):
            p, q = inter_pq_s[key]
            interval_dyn = sde.build(opt, p, q)

            if i==0:
                x1 = q.sample().to(opt.device)
                if return_evolution:
                    evolution.append(x1)
            
            ts = torch.linspace(q.time, p.time, discretisation+1).to(opt.device)
            dt = ts[1]-ts[0] #negative dtimestep

            
            for t in ts[:-1]:
                #print(x1[0])
                s_t = torch.sqrt(interval_dyn.forward_variance_accumulation(t))
                #print('s_t: ', s_t)
                x0_e = x1 - s_t*self.z_b(x1, t)
                #print('x0_e: ', x0_e)
                if torch.abs((t+dt) - p.time) <= torch.tensor(1e-6):
                    x_t_plus_dt = x0_e
                else:
                    x_t_plus_dt = interval_dyn.get_sample_from_posterior_given_pair(t+dt, x0_e, x1)
                
                #print('x_t_plus_dt: ', x_t_plus_dt)
                
                x1 = x_t_plus_dt
                if return_evolution:
                    evolution.append(x1)

        
        #print(x1[0])

        if return_evolution:
            return {'x1':x1, 'evolution':torch.stack(evolution)}
        else:
            return x1


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
