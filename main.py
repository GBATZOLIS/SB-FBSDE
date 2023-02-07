from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from runner import Runner
from MultiStageRunner import MultiStageRunner, MultistageCombiner
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
    if opt.phase == 'train':
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
            if opt.train_method == 'joint':
                run.sb_joint_train(opt)
            elif opt.train_method == 'alternate':
                run.sb_alterating_train(opt)
            
    elif opt.phase == 'test':
        if opt.training_scheme == 'divideNconquer':
            run = MultistageCombiner(opt)
            run.compute_fid()
            #run.generate_samples()

            #run = MultiStageRunner(opt)
            #run.experimental_features(opt) #-> test the effect of decreasing sigma on the ODE and SDE trajectories (plot paths, calculate average ODE curvature)

if not opt.cpu:
    with torch.cuda.device(opt.gpu):
        main(opt)
else: main(opt)
