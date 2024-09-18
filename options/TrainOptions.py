import argparse
class TrainText2MotionsOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--dataset',type=str,default='kit',help='name of dataset')
        self.parser.add_argument("--gpu_id", type=int, default=0,help='GPU id')
        self.parser.add_argument('--max_epoch',type=int,default=300,help='训练次数')
        self.parser.add_argument('--latent_dim',type=int,default=512,help='噪声向量')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--name', type=str, default="test", help='Name of this trial')
        self.parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
        self.parser.add_argument('--bert', type=str, default="true", help='is use bert')
        self.parser.add_argument("--Kt", type=int, default=1, help="kernel_size of time")
        self.parser.add_argument("--Ks", type=int, default=4, help="kernel_size of gragp") 
        self.parser.add_argument("--n_critic", type=int, default=5, help="Train the generator after n_critic discriminator steps") 

        self.parser.add_argument('--batch_size', type=int, default=4, help='每一次训练的样本数')
        self.parser.add_argument('--feat_bias', type=float, default=5, help='Layers of GRU')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='Layers of GRU')
        self.parser.add_argument('--is_continue', action="store_true", help='Training iterations')
        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress')
        self.parser.add_argument('--save_every_e', type=int, default=5, help='Frequency of printing training progress')
        self.parser.add_argument('--eval_every_e', type=int, default=3, help='Frequency of printing training progress')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of printing training progress')
        self.parser.add_argument("--unit_length", type=int, default=4, help="Length of motion")
        self.parser.add_argument("--max_text_len", type=int, default=20, help="Length of motion")
        self.parser.add_argument("--data_root", type=str, default="/root/xinglin-data/dataset", help="dataset path")
        self.parser.add_argument("--text2len", type=str, default="/root/xinglin-data/text2len/checkpoints/kit/test/model/finest.tar", help="text2len model")
    def parse(self):
        self.opt = self.parser.parse_args()
        self.opt.is_train = True
        args = vars(self.opt)
        return self.opt
