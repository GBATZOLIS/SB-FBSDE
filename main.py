from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from runner import Runner
from MultiStageRunner import MultiStageRunner
import util
import options

import colored_traceback.always
from ipdb import set_trace as debug

print(util.yellow("======================================================="))
print(util.yellow("     Likelihood-Training of Schrodinger Bridge"))
print(util.yellow("======================================================="))
print(util.magenta("setting configurations..."))
opt = options.set()

def main(opt):
    if opt.training_scheme == 'standard':
        run = Runner(opt)

        # ====== Training functions ======
        if opt.train_method=='alternate':
            run.sb_alternate_train(opt)
        elif opt.train_method=='joint':
            run.sb_joint_train(opt)

        # ====== Test functions ======
        elif opt.compute_FID:
            run.evaluate(opt, util.get_load_it(opt.load), metrics=['FID','snapshot'])
        elif opt.compute_NLL:
            run.compute_NLL(opt)
        else:
            raise RuntimeError()
    
    elif opt.training_scheme == 'divideNconquer':
        run = MultiStageRunner(opt)
        #run.sb_joint_train(opt)
        #run.sanity_check(opt) -> passed (bug solved)
        run.sb_alterating_train(opt)

if not opt.cpu:
    with torch.cuda.device(opt.gpu):
        main(opt)
else: main(opt)
