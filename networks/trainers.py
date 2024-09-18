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

        # if args.is_train:
        #     # self.motion_dis
        #     self.logger = Logger(args.log_dir)
        #     self.mul_cls_criterion = torch.nn.CrossEntropyLoss()

    def resume(self, model_dir):
        checkpoints = torch.load(model_dir, map_location=self.device)
        self.generator.load_state_dict(checkpoints['generator'])
        self.discriminator.load_state_dict(checkpoints['discriminator'])
        self.optimizer_G.load_state_dict(checkpoints['optimizer_G'])
        self.optimizer_D.load_state_dict(checkpoints['optimizer_D'])
        return checkpoints['epoch'], checkpoints['iter']

    def save(self, model_dir, epoch, niter):
        state = {
            'generator': self.generator.state_dict(),
            'discriminator':self.discriminator.state_dict(),
            'optimizer_G': self.self.optimizer_G.state_dict(),
            'optimizer_D':self.optimizer_D.state_dict(),
            'epoch': epoch,
            'niter': niter,
        }
        torch.save(state, model_dir)

    # @staticmethod
    # def zero_grad(opt_list):
    #     for opt in opt_list:
    #         opt.zero_grad()

    # @staticmethod
    # def clip_norm(network_list):
    #     for network in network_list:
    #         clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()
    def softmax(x):
        return nn.Softmax(dim=1)
    @staticmethod
    def compute_gradient_penalty(D, real_samples, fake_samples, labels):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        Tensor     = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        labels = LongTensor(labels)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates, labels)
        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.reshape(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty  
    
    def train(self, train_dataloader):
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.estimator.to(self.device)
        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))
        Tensor  = torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor
        loss_d, loss_g = [], []
        for epoch in range(self.opt.max_epoch):
            for i,motion,labels in enumerate(train_dataloader):
                word_embeddings,_,real_len,m_lens = labels
                with torch.no_grad():
                    word_emb = torch.squeeze(word_embeddings, dim=1).detach().to(self.opt.device).float()
                    pred_dis = self.estimator(word_emb, real_len)
                    pred_dis = self.softmax(pred_dis).cpu().numpy()
                    predicted_lens = np.argmax(pred_dis,axis =1)
                t_size = predicted_lens*self.opt.unit_length

                # train discriminator
                self.optimizer_D.zero_grad()

                z = Variable(Tensor(np.random.normal(0, 1, (self.opt.batch_size, self.opt.latent_dim))))

                # generator
                '''
                    z:tensor (batch_size,latent_dim)
                    word_embeddings:tensor (batch_size,word_dim=3072)
                    t_size:list [batch_size]
                '''
                fake_motion = self.generator(z,word_embeddings,t_size)

                #real actions 
                real_validity = self.discriminator(motion,labels)
                fake_validity = self.discriminator(fake_motion, labels)
                # Gradient penalty
                gradient_penalty = self.compute_gradient_penalty(self.discriminator, motion.data, fake_motion.data, labels.data)
                #Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.opt.lambda_gp * gradient_penalty
                d_loss.backwaed()
                self.optimizer_D.step()

                self.optimizer_G.zero_grad()
                if i % self.opt.n_critic == 0:
                    '''
                        train generator
                    '''
                    fake_motion = self.generator(z,word_embeddings,t_size)
                    fake_validity = self.discriminator(fake_motion, labels)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    self.optimizer_G.step()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, self.opt.max_epochs, i, len(train_dataloader), d_loss.item(), g_loss.item())
                )
                loss_d.append(d_loss.data.cpu())
                loss_g.append(g_loss.data.cpu())
                
            if epoch % self.opt.save_every_e == 0:
                self.save(self.opt.save_root)



                
            
