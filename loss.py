
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

