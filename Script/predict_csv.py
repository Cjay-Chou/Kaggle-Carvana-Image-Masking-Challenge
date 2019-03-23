import cv2
import numpy as np
import pandas as pd
import threading
import queue
import tensorflow as tf
from tqdm import tqdm
from keras.models import load_model
from model import  bce_dice_loss, dice_coef


MODEL_WIDTH = 1024
MODEL_HEIGHT = 1024
BATCH_SIZE = 10

ORIG_WIDTH = 1918
ORIG_HEIGHT = 1280

def length_encode(mask):
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle

test_csv = pd.read_csv('../input/sample_submission.csv')
test_list = list(test_csv['img'])
result_list = []

model = load_model(
    filepath='logs/model_weights.hdf5',
    custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coef': dice_coef}
)

#graph = tf.get_default_graph()

for start in tqdm(range(0, len(test_list), BATCH_SIZE)):
    end = min(start + BATCH_SIZE, len(test_list)) #edge security
    test_fname_batch = test_list[start:end]
    test_img_batch = []
    for fname in test_fname_batch:
        test_img = cv2.imread('../input/test/{}'.format(fname))
        test_img = cv2.resize(test_img, (MODEL_WIDTH, MODEL_HEIGHT))

        test_img_batch.append(test_img)

    input_batch = np.array(test_img_batch)

    result_batch = model.predict_on_batch(input_batch)
    result_batch = np.squeeze(result_batch, axis=3)

    for result in result_batch:
        result_orgsize = cv2.resize(result, (ORIG_WIDTH, ORIG_HEIGHT))
        mask = result_orgsize > 0.5
        result_coded = length_encode(mask)
        result_list.append(result_coded)
    
    print(start,'/',len(test_list))


df = pd.DataFrame({'img': test_list, 'rle_mask': result_list})
df.to_csv('submission2.csv.gz', index=False, compression='gzip')