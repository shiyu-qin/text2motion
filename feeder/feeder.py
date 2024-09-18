import torch
from tqdm import tqdm
import os
import numpy as np
from os.path import join as pjoin
def collate_fn(batch):
    '''
    参数:
        batch (list of tuples): 批次中的样本，每个样本是一个元组
        (word_embeeding)
        self.word_ems[index],self.masks[index],self.labels[index],self.motions[index]
    '''
    word_embeddings_batch = [item[0] for item in batch]
    masks_batch = [item[1] for item in batch]
    labels_batch = [item[2] for item in batch]
    motions_batch = [item[3] for item in batch]
    real_len_batch = [item[4] for item in batch]
    # m_len_batch = torch.tensor(m_len_batch)
    return torch.stack(word_embeddings_batch,dim=0), masks_batch, labels_batch,motions_batch,torch.tensor(real_len_batch)

class Feeder(torch.utils.data.Dataset):
    def __init__(self,opt,data_path):
        self.opt = opt
        self.max_length = 20
        min_motion_len = 24 if self.opt.dataset =='kit' else 24
        # print("data_path",data_path)
        filenames = []
        with open(data_path, 'r') as file:
            for line in file:
                filenames.append(line.strip())
        # print(filenames)
        self.labels = []
        self.motions_len = []
        self.word_ems = []
        self.masks = []
        self.motions = []
        self.real_len = []
        for filename in tqdm(filenames):
            motion_path = pjoin(opt.motion_dir,filename + '.npy')
            if not os.path.exists(motion_path):
                continue
            motion = np.load(motion_path)
            if(len(motion) < min_motion_len or len(motion) >= 200 ):
                continue
            # print(motion_path)
            if filename[0] == 'M':
                filename = filename[1:]
            label_path = pjoin(opt.text_dir,filename+'.npy')
            if not os.path.exists(label_path):
                # print("no exist")
                continue
            # print(label_path)
            # print(motion_path,label_path)
            label = np.load(label_path,allow_pickle=True).item()
            word_em = label['embeddings']
            mask = label['mask']
            self.word_ems.append(word_em)
            self.masks.append(mask)
            self.real_len.append(torch.sum(mask == 1,dim = 1)[0])
            self.labels.append(label)
            self.motions_len.append(motion.shape[0])
            self.motions.append(motion)
        print("len(self.motions)",len(self.motions))
    def __len__(self):
        return len(self.motions_len)
    def __getitem__(self, index):
        '''
        return:
            :word_ems
            :masks
            :labels
            :motions
        '''
        return self.word_ems[index],self.masks[index],self.labels[index],self.motions[index],self.real_len[index]

