import keras.backend as K
import numpy as np
import random
import matplotlib.pyplot as plt

def plot_random_img(data):
    images, labels = data.next()
    random_number = random.randint(0, 16)
    plt.imshow(images[random_number])
    plt.title(f"Original image")
    plt.axis(False)

def plot_loss_curves(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    val_loss = [i*100. for i in val_loss]
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.figure(figsize=(4,4))
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.figure(figsize=(4,4))
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

def rle2mask(rle):
    if rle==0:
        return np.zeros((128, 800), dtype=np.uint8)
    
    height = 256
    width = 1600
    mask = np.zeros(width*height, dtype=np.uint8)
    
    starts_lengths = np.asarray([int(x) for x in rle.split()])
    starts = starts_lengths[0::2]-1 
    lengths = starts_lengths[1::2]
    for idx, start in enumerate(starts): 
        mask[int(start): int(start+lengths[idx])] = 1
        
    return mask.reshape((height, width), order='F')[::2, ::2]

def mask2contour(mask, width=3):
    w = mask.shape[1]
    h = mask.shape[0]
    
    mask2 = np.concatenate([mask[:, width:], np.zeros((h, width))], axis=1)
    mask2 = np.logical_xor(mask, mask2)
    mask3 = np.concatenate([mask[width:, :], np.zeros((width, w))], axis=0)
    mask3 = np.logical_xor(mask, mask3)
    
    return np.logical_or(mask2, mask3)

def mask_padding(mask, pad=2):
    w = mask.shape[1]
    h = mask.shape[0]
    
    for i in range(1, pad, 2):
        temp = np.concatenate([mask[i:, :], np.zeros((i, w))], axis=0)
        mask = np.logical_or(mask, temp)

    for i in range(1, pad, 2):
        temp = np.concatenate([np.zeros((i, w)), mask[:-i, :]], axis=0)
        mask = np.logical_or(mask, temp)
   
    for i in range(1, pad, 2):
        temp = np.concatenate([mask[:, i:], np.zeros((h, i))], axis=1)
        mask = np.logical_or(mask, temp)

    for i in range(1, pad, 2):
        temp = np.concatenate([np.zeros((h, i)), mask[:, :-i]], axis=1)
        mask = np.logical_or(mask, temp)
    
    return mask

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f* y_pred_f)
    return (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)