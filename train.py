import os
import keras
import numpy as np
import pandas as pd
import cv2
import random
import matplotlib.pylab as plt
from model import get_net
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

WIDTH = 1024
HEIGHT = 1024
BATCH_SIZE = 2

def length_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def length_encode(mask):
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


def get_image(image_id, image_type):
    if "image" == image_type:
        fname = "../input/train/{}".format(image_id)
        img = cv2.imread(fname)
        if img is None:
            print("image read faild:",fname)
        return img
    if "mask" == image_type:
        number = image_list.index(image_id)
        mask = length_decode(mask_list[number],(1280,1918))
        return mask

def train_test_split(ids_train):
    car_id = ids_train['img'].apply(lambda x: x[:-7])
    car_id = car_id.unique().tolist()
    test_list = []
    train_list = []
    #--- add 18 car in test_list
    for i in range(18):
        pick_car = random.choice(car_id)
        car_id.remove(pick_car)
        for j in range(16):
            test_list.append(pick_car+'_{:0>2d}.jpg'.format(j+1))

    #--- random add every car 2 direction to test_list
    for each_id in car_id:

        random_id = random.sample(range(16),2)
        for num in range(16):
            if num in random_id:
                test_list.append(each_id+'_{:0>2d}.jpg'.format(num+1))
            else:
                train_list.append(each_id+'_{:0>2d}.jpg'.format(num+1))
    
    return train_list, test_list


def batch_generator(fnamelist, batch_size):
    while True:
        shuffle_indices = list(range(len(fnamelist)))
        random.shuffle(shuffle_indices)
        
        for i in range(0, len(shuffle_indices),batch_size):
            x_batch = []
            y_batch = []
            
            for idx in shuffle_indices[i:i+batch_size]:
                image = get_image(fnamelist[idx],"image")
                image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                mask = get_image(fnamelist[idx],"mask")
                mask = cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                mask = np.expand_dims(mask, axis=-1)
                x_batch.append(image)
                y_batch.append(mask)
                
            yield np.array(x_batch), np.array(y_batch)

if __name__ == '__main__':
    train_masks_csv = pd.read_csv('../input/train_masks.csv')
    image_list = list(train_masks_csv['img'])
    mask_list = list(train_masks_csv["rle_mask"])

    ids_train, ids_valid = train_test_split(train_masks_csv)

    model = get_net(input_shape=(HEIGHT, WIDTH, 3))

    callbacks = [EarlyStopping(monitor='val_dice_coef',
                               patience=10,
                               verbose=1,
                               min_delta=1e-4,
                               mode='max'),
                 ReduceLROnPlateau(monitor='val_dice_coef',
                                   factor=0.2,
                                   patience=5,
                                   verbose=1,
                                   epsilon=1e-4,
                                   mode='max'),
                 ModelCheckpoint(monitor='val_dice_coef',
                                 filepath='logs/model_weights.hdf5',
                                 save_best_only=True,
                                 mode='max'),
                 TensorBoard(log_dir='logs')]

    model.fit_generator(generator=batch_generator(ids_train, BATCH_SIZE),
                        steps_per_epoch=np.ceil(float(len(ids_train)) / float(BATCH_SIZE)),
                        epochs=100,
                        #verbose=2,
                        callbacks=callbacks,
                        validation_data=batch_generator(ids_valid, BATCH_SIZE),
                        validation_steps=np.ceil(float(len(ids_valid)) / float(BATCH_SIZE)))

    keras.backend.clear_session()