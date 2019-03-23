import keras.backend as K #Using backend will make it faster to calculate
from keras.losses import binary_crossentropy
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, add, concatenate, Dropout
from keras.models import Model
from keras.optimizers import RMSprop

#==================loss=====================
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

#=====================model=============================
def CreateConvBlock(x, filters , n = 2, name = 'convblock'):
    for i in range(n):
        x = Conv2D(filters[i], (3,3), activation='relu', padding='same',name=name+'_conv'+str(i+1))(x)
    convresult = x
    x = MaxPooling2D(pool_size=(2,2),strides=(2, 2) ,name=name+'_pooling')(x)
    return x, convresult

#Origin, three times dilation conv at the bottom，and add them in the end.
def BottleBlock(x, filters,depth=3,
               activation='relu',name = 'Bottle'):
    dilated_layers = []
    for i in range(depth):
        x = Conv2D(filters, (3,3),
                   activation='relu', padding='same', dilation_rate=2**i)(x)
        dilated_layers.append(x)
    return add(dilated_layers, name = name+'_add')

def CreateUpConvBlock(x, contractpart, filters, n = 2, name = 'upconvblock'):
    x = UpSampling2D(size=(2, 2),name = name+'upsample')(x)
    x = Conv2D(filters[0], (3,3), activation='relu', padding='same',name = name+'conv_0')(x)
    x = concatenate([contractpart, x],name = name+'_concat')
    for i in range(n):
        x = Conv2D(filters[i], (3,3), activation='relu', padding='same',name = name+'conv_'+str(i+1))(x)
    return x

def ConstructUnetModel(input, nclass = 1, use_dropout = True):
    x, contract1 = CreateConvBlock(input, (32, 32), n = 2, name = 'contract1')
    x, contract2 = CreateConvBlock(x, (64, 64), n = 2, name = 'contract2')
    x, contract3 = CreateConvBlock(x, (128, 128), n = 2, name = 'contract3')
    x = BottleBlock(x, 256, depth = 3, name = 'Bottle')
    if use_dropout:
        x = Dropout(0.3, name='dropout')(x)
    x = CreateUpConvBlock(x, contract3, (128, 128), n = 2,name = 'expand3')
    x = CreateUpConvBlock(x, contract2, (64, 64), n = 2,name = 'expand2')
    x = CreateUpConvBlock(x, contract1, (32, 32), n = 2,name = 'expand1')
    #last classify
    x = Conv2D(nclass, (1, 1), activation='sigmoid',name='classify')(x)
    
    return x

#====================Build===============================

def get_net(input_shape=(1280, 1920, 3), lr = 0.0001):
    inputs = Input(shape=input_shape, name="input")
    classify = ConstructUnetModel(inputs)
    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=RMSprop(lr), loss=bce_dice_loss, metrics=[dice_coef])
    
    return model
