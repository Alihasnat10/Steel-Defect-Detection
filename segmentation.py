import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers
from tf_data_loader import DataLoader
from utils import dice_coef, mask2contour, mask_padding

class Segmentation:
    def __init__(self, data_root_path):
        self.data_root_path = data_root_path

    def generate_data(self):
        train = pd.read_csv(self.data_root_path + 'train.csv')
        images = pd.DataFrame({'ImageId': os.listdir(self.data_root_path+'train_images')})
        train2 = pd.merge(train, images, how='right').sort_values('ImageId')
        train2 = train2.reset_index(drop=True)

        train2 = train2.pivot(index=['ImageId'], columns=['ClassId'], values=['EncodedPixels'])
        train2 = train2.reset_index()

        train2 = train2.groupby('ImageId').sum().reset_index()
        train2.columns = ['ImageId', ' ', 'e1', 'e2', 'e3', 'e4']
        train2 = train2.drop(' ', axis=1)

        train2['count'] = train2.apply(lambda x: 4-x.value_counts()[0], axis=1)
        return train2

    def get_model(self, input_shape, init_node, classify):
        out_dim = init_node
        
        input_ = tf.keras.Input(shape=input_shape)
        
        # Contracting Path
        conv1 = layers.Conv2D(out_dim, 3, padding='same', activation='relu')(input_)
        conv1_out = layers.Conv2D(out_dim, 3, padding='same', activation='relu')(conv1)
        conv2 = layers.MaxPool2D()(conv1_out)
        
        out_dim *= 2
        conv2 = layers.Conv2D(out_dim, 3, padding='same', activation='relu')(conv2)
        conv2_out = layers.Conv2D(out_dim, 3, padding='same', activation='relu')(conv2)
        conv3 = layers.MaxPool2D()(conv2_out)
        
        out_dim *= 2
        conv3 = layers.Conv2D(out_dim, 3, padding='same', activation='relu')(conv3)
        conv3_out = layers.Conv2D(out_dim, 3, padding='same', activation='relu')(conv3)
        conv4 = layers.MaxPool2D()(conv3_out)
        
        out_dim *= 2
        conv4 = layers.Conv2D(out_dim, 3, padding='same', activation='relu')(conv4)
        conv4_out = layers.Conv2D(out_dim, 3, padding='same', activation='relu')(conv4)
        conv5 = layers.MaxPool2D()(conv4_out)
        
        # Mid
        out_dim *= 2
        conv_mid = layers.Conv2D(out_dim, 3, padding='same', activation='relu')(conv5)
        conv_mid = layers.Conv2D(out_dim, 3, padding='same', activation='relu')(conv_mid)
        
        # Expansive Path
        out_dim /= 2
        t_conv1 = layers.Conv2DTranspose(out_dim, 3, strides=2, padding='same', activation='relu')(conv_mid)
        t_conv1 = layers.Concatenate()([conv4_out, t_conv1])
        t_conv1 = layers.Conv2D(out_dim, 3, padding='same', activation='relu')(t_conv1)
        t_conv1 = layers.Conv2D(out_dim, 3, padding='same', activation='relu')(t_conv1)
        
        out_dim /= 2
        t_conv2 = layers.Conv2DTranspose(out_dim, 3, strides=2, padding='same', activation='relu')(t_conv1)
        t_conv2 = layers.Concatenate()([conv3_out, t_conv2])
        t_conv2 = layers.Conv2D(out_dim, 3, padding='same', activation='relu')(t_conv2)
        t_conv2 = layers.Conv2D(out_dim, 3, padding='same', activation='relu')(t_conv2)
        
        out_dim /= 2
        t_conv3 = layers.Conv2DTranspose(out_dim, 3, strides=2, padding='same', activation='relu')(t_conv2)
        t_conv3 = layers.Concatenate()([conv2_out, t_conv3])
        t_conv3 = layers.Conv2D(out_dim, 3, padding='same', activation='relu')(t_conv3)
        t_conv3 = layers.Conv2D(out_dim, 3, padding='same', activation='relu')(t_conv3)
        
        out_dim /= 2
        t_conv4 = layers.Conv2DTranspose(out_dim, 3, strides=2, padding='same', activation='relu')(t_conv3)
        t_conv4 = layers.Concatenate()([conv1_out, t_conv4])
        t_conv4 = layers.Conv2D(out_dim, 3, padding='same', activation='relu')(t_conv4)
        t_conv4 = layers.Conv2D(out_dim, 3, padding='same', activation='relu')(t_conv4)
        
        output = layers.Conv2D(classify, 1, padding='same', activation='sigmoid')(t_conv4)
        
        model = tf.keras.Model(inputs=[input_], outputs=[output])
        
        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=dice_coef)
    
        return model
    
    def train(self, train_loader, valid_loader, model_input_shape, batch_size, classify_nodes):
        model = self. get_model(model_input_shape, batch_size, classify_nodes)
        
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('defect_segmentation.hdf5',
                                                   monitor='val_dice_coef',
                                                   mode='max',
                                                   save_best_only=True,
                                                   save_weights_only=True)


        history = model.fit_generator(train_loader,
                                    validation_data=valid_loader,
                                    epochs=5,
                                    callbacks=[checkpoint_cb])

        return history, model
    
    def visualize_data(self, train):
        defects = list(train[train['e1']!=0].sample(4).index)
        defects += list(train[train['e2']!=0].sample(4).index)
        defects += list(train[train['e3']!=0].sample(4).index)
        defects += list(train[train['e4']!=0].sample(4).index)

        train_batches = DataLoader(train[train.index.isin(defects)], self.data_root_path, shuffle=True)
        print('defect1: red, defect2: green, defect3: yellow, defect4: purple')

        plt.figure(figsize=(8, 30))
        for batch in train_batches:
            for i in range(16):
                plt.subplot(16, 1, i+1)
                img = batch[0][i]/255
                defect = []
                for j in range(4):
                    msk = batch[1][i, ..., j]
                    msk = mask_padding(msk)
                    msk = mask2contour(msk)
                    if np.sum(msk) != 0:
                        defect.append(j+1)
                    if j == 0:
                        img[msk==1] = 1, 0, 0
                    elif j == 1:
                        img[msk==1] = 0, 1, 0
                    elif j == 2:
                        img[msk==1] = 1, 1, 0
                    elif j == 3:
                        img[msk==1] = 1, 0, 1
                        
                plt.title(f'defect: {defect}')
                plt.axis('off')
                plt.imshow(img)
            plt.show()

if __name__ == "__main__":
    root = 'data/'
    visualize_data = False
    segment = Segmentation(root)
    train = segment.generate_data()

    if visualize_data:
        segment.visualize_data(train)

    train_idx = int(0.8*len(train))
    train_loader = DataLoader(train.iloc[:train_idx], root, shuffle=True)
    valid_loader = DataLoader(train.iloc[train_idx:], root)

    history, model = segment.train(train_loader, valid_loader, (128, 800, 3), 16, 4)

    result = model.evaluate(valid_loader)