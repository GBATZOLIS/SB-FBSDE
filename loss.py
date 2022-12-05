import math
import torch
import util
from ipdb import set_trace as debug

def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y)


def sample_e(opt, x):
    return {
        'gaussian': sample_gaussian_like,
        'rademacher': sample_rademacher_like,
    }.get(opt.noise_type)(x)


def compute_div_gz(opt, dyn, ts, xs, policy, return_zs=False):

    zs = policy(xs,ts)

    g_ts = dyn.g(ts)
    g_ts = g_ts[:,None,None,None] if util.is_image_dataset(opt) else g_ts[:,None]
    gzs = g_ts*zs

    e = sample_e(opt, xs)
    e_dzdx = torch.autograd.grad(gzs, xs, e, create_graph=True)[0]
    
    div_gz = e_dzdx * e
    # approx_div_gz = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)

    return [div_gz, zs] if return_zs else div_gz


def compute_sb_nll_alternate_train(opt, dyn, ts, xs, zs_impt, policy_opt, return_z=False):
    """ Implementation of Eq (18,19) in our main paper.
    """
    assert opt.train_method == 'alternate'
    assert xs.requires_grad
    assert not zs_impt.requires_grad

    #batch_x = opt.train_bs_x
    #batch_t = opt.train_bs_t
    batch_x_times_batch_t = ts.size(0)

    with torch.enable_grad():
        div_gz, zs = compute_div_gz(opt, dyn, ts, xs, policy_opt, return_zs=True)
        loss = zs*(0.5*zs + zs_impt) + div_gz
        #loss = torch.sum(loss * dyn.dt) / batch_x / batch_t  # sum over x_dim and T, mean over batch
        loss = torch.sum(loss * dyn.dt) / batch_x_times_batch_t  # sum over x_dim and T, mean over batch
    
    
    return loss, zs if return_z else loss

def compute_sb_nll_joint_increment(opt, dyn, ts, xs_f, zs_f, policy_b, x_term_f, orig_x):
    #x_term_f is None for all levels and intervals apart from the last interval of the last level. last_level_last_stage=True.
    assert xs_f.requires_grad and zs_f.requires_grad
    if x_term_f is not None:
        assert x_term_f.requires_grad

    if orig_x is not None:
        def get_loglikelihood_approx_fn(points, alpha, sigma):
            def loglikelihood_approx_fn(x):
                N = torch.tensor(points.size(0)) #number of datapoints
                d = x.size(1) #dimension
                exps = torch.ones((x.size(0), points.size(0)), device=x.device)
                for i in range(points.size(0)):
                    reduce_dims = tuple([i+1 for i in range(len(x.shape)-1)])
                    exps[:, i] = -0.5*torch.sum((x - points[i] * alpha)**2, dim=reduce_dims)/sigma**2
                return -torch.log(N)-d/2*torch.log(2*math.pi*sigma)+torch.logsumexp(exps, dim=1)
            return loglikelihood_approx_fn
        #alpha, sigma = dyn.q.get_perturbation_kernel()
        alpha = torch.tensor(0.)
        sigma = torch.tensor(1.)
        #print(alpha, sigma)
        loglikelihood_approx_fn = get_loglikelihood_approx_fn(orig_x, alpha, sigma)
    else:
        def get_loglikelihood_approx_fn(dyn):
            def loglikelihood_approx_fn(x):
                return dyn.q.log_prob(x)
            return loglikelihood_approx_fn
        loglikelihood_approx_fn = get_loglikelihood_approx_fn(dyn)

    batch_x_times_batch_t = ts.size(0)
    with torch.enable_grad():
        div_gz_b, zs_b = compute_div_gz(opt, dyn, ts, xs_f, policy_b, return_zs=True)
        loss = 0.5*(zs_f + zs_b)**2 + div_gz_b
        loss = torch.sum(loss*dyn.dt) / batch_x_times_batch_t
        
        if x_term_f is not None:
            avg_loglikelihood = loglikelihood_approx_fn(x_term_f).mean()
            print(avg_loglikelihood)
            loss -= avg_loglikelihood
    
    return loss

def compute_sb_nll_joint_train(opt, batch_x, dyn, ts, xs_f, zs_f, x_term_f, policy_b):
    """ Implementation of Eq (16) in our main paper.
    """
    assert opt.train_method == 'joint'
    assert policy_b.direction == 'backward'
    assert xs_f.requires_grad and zs_f.requires_grad and x_term_f.requires_grad

    div_gz_b, zs_b = compute_div_gz(opt, dyn, ts, xs_f, policy_b, return_zs=True)
    loss = 0.5*(zs_f + zs_b)**2 + div_gz_b
    loss = torch.sum(loss*dyn.dt) / batch_x
    loss = loss - dyn.q.log_prob(x_term_f).mean()
    return loss

