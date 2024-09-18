import torch
import torch.nn as nn 
import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd
class Text2MotionTrainer(object):
    def __init__(self, args, generator,discriminator,estimator):
        self.opt = args
        self.generator = generator
        self.estimator = estimator 
        self.discriminator = discriminator
        self.device = args.device

        if args.is_train:
            # self.motion_dis
            self.logger = Logger(args.log_dir)
            self.mul_cls_criterion = torch.nn.CrossEntropyLoss()