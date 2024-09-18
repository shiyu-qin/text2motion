from options.TrainOptions import TrainText2MotionsOptions
from networks.trainers import Text2MotionTrainer
from networks.generator import Generator,MotionLenEstimator_2
from networks.discriminator import Discriminator
from os.path import join as pjoin
from sklearn.model_selection import train_test_split
from utils.trainUtils import splitDataset
from torch.utils.data import DataLoader,random_split
from feeder.feeder import Feeder,collate_fn
import torch
import torch.nn as nn 
import numpy as np
from torch.autograd import Variable
# from text2Len. import
def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
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

def pre_lens(text2len,word_emb,real_len):
    checkpoints = torch.load(text2len,weights_only=True)
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        if opt.bert == 'true':
            estimator = MotionLenEstimator_2(3072, 512,200 //4)
            estimator.load_state_dict(checkpoints['estimator'])
            estimator.to(opt.device)
            pred_dis = estimator(word_emb, real_len)
        pred_dis = softmax(pred_dis).cpu().numpy()
        predicted_lens = np.argmax(pred_dis,axis =1 )
    return predicted_lens
def save(self, model_dir, generator,epoch, niter):
    state = {
        'generator': generator.state_dict(),
        'discriminator':discriminator.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
        'optimizer_D':optimizer_D.state_dict(),
        'epoch': epoch,
        'niter': niter,
    }
    torch.save(state, model_dir)

if __name__=='__main__':
    print("start=========================")
    parser = TrainText2MotionsOptions()
    opt = parser.parse()
    opt.device = torch.device('cpu' if opt.gpu_id==-1 else 'cuda:'+str(opt.gpu_id))
    # print("device",opt.device)
    torch.autograd.set_detect_anomaly(True)
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset, opt.name)
    opt.model_dir = pjoin(opt.save_root,'model')
    if opt.dataset == 'kit':
        # opt.data_root = '../dataset/kit'
        opt.motion_dir = pjoin(opt.data_root,'new_joints')
        # print("opt.motion_dir",opt.motion_dir)
        if opt.bert =='true':
            opt.text_dir = pjoin(opt.data_root,'texts2b')
            dim_word = 3072
        else:
            opt.text_dir = pjoin(opt.data_root,'text')
            dim_word = 300
        opt.joints_num = 21
        fps = 12.5
        radius = 240 * 8
        dim_pose = 251
        opt.latent_dim = 512
    else:
        raise KeyError('Dataset Does Not Exist')
    train_split_file, test_split_file = splitDataset(opt.data_root,opt.dataset)
    train_dataset = Feeder(opt,train_split_file)
    train_loader = DataLoader(train_dataset,batch_size = opt.batch_size,shuffle=True,collate_fn=collate_fn)

    generator = Generator(opt.latent_dim, out_channels=3, n_classes=dim_word, t_size=64)
    discriminator = Discriminator(in_channels=3, out_channels=2,Kt = opt.Kt,Ks = opt.Ks)

    generator.to(opt.device)
    discriminator.to(opt.device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    cuda = torch.cuda.is_available()
    Tensor     = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    min_loss_d = 0x3ffff
    min_loss_g = 0x3ffff
    for epoch in range(opt.max_epoch):
        for i,batch_data in enumerate(train_loader):
            word_embeddings,masks,labels,motions,real_len = batch_data
            # word_emb = word_ems.detach().to(opt.device).float()
            word_emb = torch.squeeze(word_embeddings, dim=1).detach().to(opt.device).float()
            real_len = real_len.to(opt.device).float()
            pre_len = pre_lens(opt.text2len,word_emb,real_len)*4
            print("pre_len",pre_len,type(pre_len))#[9,17,22,15]
            print("word_emb",word_emb.size())#[bs,50,3072]
            optimizer_D.zero_grad()
            z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))
            fake_imgs = generator(z, word_emb,pre_len)

            # Real actions
            real_validity = discriminator(motions, labels)
            # Fake actions
            fake_validity = discriminator(fake_imgs, labels)
            
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, motions.data, fake_imgs.data, labels.data)
            
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + opt.lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            
            # Train the generator after n_critic discriminator steps
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of actions
                fake_imgs = generator(z, labels)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake actions
                fake_validity = discriminator(fake_imgs, labels)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            loss_d.append(d_loss.data.cpu())
            loss_g.append(g_loss.data.cpu())
            if epoch % opt.save_every_e==0:
                    state = {
                        'generator': generator.state_dict(),
                        'discriminator':discriminator.state_dict(),
                        'optimizer_G': optimizer_G.state_dict(),
                        'optimizer_D':optimizer_D.state_dict(),
                        'epoch': epoch
                    }
                    torch.save(state, os.path.join(opt.model_dir, "mdoel_%d.pth" % epoch))
            if d_loss.data < min_loss_d and g_loss.data < min_loss_g :
                min_loss_d = d_loss.data
                min_loss_g = g_loss.data
                state = {
                    'generator': generator.state_dict(),
                    'discriminator':discriminator.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'optimizer_D':optimizer_D.state_dict(),
                    'epoch': epoch
                }
                torch.save(state, os.path.join(opt.model_dir, "finest.pth"))
            # break
        # break
    state = {
        'generator': generator.state_dict(),
        'discriminator':discriminator.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
        'optimizer_D':optimizer_D.state_dict(),
        'epoch': epoch}
    torch.save(state, os.path.join(opt.model_dir, "lastest%d.pth" % epoch))    

    # generator = Generator(opt.latent_dim, opt.out_channels, opt.dim_word, t_size=64)
    # discriminator