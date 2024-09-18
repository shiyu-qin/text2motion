from os.path import join as pjoin
from sklearn.model_selection import train_test_split

def splitDataset(data_root,dataset):
    if dataset == 'kit':
        split_file = pjoin(data_root, 'all.txt')
        filenames = []
        with open(split_file, 'r') as file:
            for line in file:
                filenames.append(line.strip())
        # train_split_file = pjoin(opt.data_root, 'train.txt')
        # val_split_file = pjoin(data_root, 'val.txt')
        train_split_file, test_split_file = train_test_split(filenames, test_size=0.3, random_state=42)
        train_path = pjoin(data_root, 'train.txt')
        test_split = pjoin(data_root, 'test.txt')
        with open(train_path, 'w') as file:
            for filename in train_split_file:
                file.write(f"{filename}\n")
        with open(test_split, 'w') as file:
            for filename in test_split_file:
                file.write(f"{filename}\n")
    return train_path,test_split