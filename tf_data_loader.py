import math 
import numpy as np
from PIL import Image
import tensorflow as tf
from utils import rle2mask

class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, df, path, batch_size=16, subset='train', shuffle=False):
        super().__init__()
        self.df = df
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        
        if self.subset == 'train':
            self.data_path = path + 'train_images/'
        elif self.subset == 'test':
            self.data_path = path + 'test_images/'
        self.on_epoch_end()
       
    def __len__(self):
        return math.ceil(len(self.df)/self.batch_size)
    
    def __getitem__(self, idx):
        X = np.zeros((self.batch_size, 128, 800, 3), dtype=np.float32)
        Y = np.zeros((self.batch_size, 128, 800, 4), dtype=np.int8)
        indices = self.indices[idx*self.batch_size: (idx+1)*self.batch_size]
        
        for i, f in enumerate(self.df['ImageId'].iloc[indices]):
            X[i] = Image.open(self.data_path + f).resize((800, 128))
            if self.subset == 'train':
                for j in range(4):
                    Y[i, ..., j] = rle2mask(self.df['e'+str(j+1)].iloc[indices[i]])
                
        if self.subset == 'train':
            return X, Y
        else:
            return X
        
    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indices)