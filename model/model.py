import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Input, Embedding, LSTM, Dense, merge, Convolution2D, GRU, TimeDistributedDense, Reshape, \
    MaxPooling2D, Convolution1D, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import random as rn
import pandas as pd
import tensorflow as tf

np.random.seed(1)
rn.seed(1)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
import keras
from keras import backend as K

tf.set_random_seed(0)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)



def plotLoss(history):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.legend(['trainloss', 'valloss'], loc='upper left')
    plt.savefig("figures/" + 'lstmloss06' + ".png", dpi=600, facecolor='w', edgecolor='w', orientation='portrait',
                papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None)
    plt.close()

    ## PLOT CINDEX
    plt.figure()
    plt.title('model  accuracy')
    plt.ylabel('Q8 accuracy')
    plt.xlabel('epoch')
    plt.plot(history.history['weighted_accuracy'])
    plt.plot(history.history['val_weighted_accuracy'])
    plt.legend(['trainaccuracy', 'valaccuracy'], loc='upper left')

    plt.savefig("figures/" + 'lstmaccuracy06' + ".png", dpi=600, facecolor='w', edgecolor='w', orientation='portrait',
                papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None)


def build_model():
    auxiliary_input = Input(shape=(700, 21), name='aux_input')  # 24
    # auxiliary_input = Masking(mask_value=0)(auxiliary_input)
    concat = auxiliary_input

    conv1_features = Convolution1D(42, 1, activation='relu', border_mode='same', W_regularizer=l2(0.001))(concat)
    # print 'conv1_features shape', conv1_features.get_shape()
    conv1_features = Reshape((700, 42, 1))(conv1_features)

    conv2_features = Convolution2D(42, 3, 1, activation='relu', border_mode='same', W_regularizer=l2(0.001))(
        conv1_features)
    # print 'conv2_features.get_shape()', conv2_features.get_shape()

    conv2_features = Reshape((700, 42 * 42))(conv2_features)
    conv2_features = Dropout(0.5)(conv2_features)
    conv2_features = Dense(400, activation='relu')(conv2_features)

    # , activation='tanh', inner_activation='sigmoid',dropout_W=0.5,dropout_U=0.5
    lstm_f1 = LSTM(output_dim=300, return_sequences=True, inner_activation='sigmoid', dropout_W=0.5, dropout_U=0.5)(
        conv2_features)
    lstm_b1 = LSTM(output_dim=300, return_sequences=True, go_backwards=True, inner_activation='sigmoid', dropout_W=0.5,
                   dropout_U=0.5)(conv2_features)

    lstm_f2 = LSTM(output_dim=300, return_sequences=True, inner_activation='sigmoid', dropout_W=0.5, dropout_U=0.5)(
        lstm_f1)
    lstm_b2 = LSTM(output_dim=300, return_sequences=True, go_backwards=True, inner_activation='sigmoid', dropout_W=0.5,
                   dropout_U=0.5)(lstm_b1)

    concat_features = merge([lstm_f2, lstm_b2, conv2_features], mode='concat', concat_axis=-1)

    concat_features = Dropout(0.4)(concat_features)
    protein_features = Dense(600, activation='relu')(concat_features)
    # protein_features = TimeDistributedDense(600,activation='relu')(concat_features)
    # protein_features = TimeDistributedDense(100,activation='relu', W_regularizer=l2(0.001))(protein_features)
    aux_output = TimeDistributedDense(1, activation='softmax', name='aux_output')(protein_features)