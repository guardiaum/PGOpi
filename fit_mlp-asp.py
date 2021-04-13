#!/usr/bin/env python
# coding: utf-8

from os import listdir, makedirs
from os.path import isfile, join, exists
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors, Word2Vec, FastText
from util import *
import time
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
from numpy import load
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Bidirectional, Attention, TimeDistributed, Conv1D, Embedding, MaxPooling1D
from tensorflow.keras.layers import Lambda, Activation, Dropout, Flatten, Concatenate, AveragePooling1D
from tensorflow.keras.layers import Multiply, Subtract
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import plot_model
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score, classification_report
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, randint

#tf.config.set_visible_devices([], 'GPU')

gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))

if gpus:
    # Create 2 virtual GPUs with 1GB memory each
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

    
"""# **Auxiliary functions**"""

def my_args():
    
    with_other = True   #True/False
    sim_value = "5"
    product_name = "cells"   # cells, cameras, dvds, routers, laptops
    product_name_ = "cel"  # cel, cam, dvd, rou, lap

    root = "./"
    datasets_dir = root + "datasets/"
    
    model_name, processed_datasets, label_encoder_path, modelcheckpoints = '', '', '', ''
    if with_other==True:
        processed_datasets = datasets_dir + "data_preprocessed/with_other/{0}/{1}/".format(product_name, sim_value)
        test_path = "./datasets/data_preprocessed/with_other/{0}/test_sets/".format(product_name)
        modelcheckpoints = root + "models/checkpoints/with_other/{0}/{1}/".format(product_name, sim_value)
        model_name = "BLSTM+MLP-aspect_with-other"
    else:
        processed_datasets = datasets_dir + "data_preprocessed/no_other/{0}/{1}/".format(product_name, sim_value)
        test_path = "./datasets/data_preprocessed/no_other/{0}/test_sets/".format(product_name)
        modelcheckpoints = root + "models/checkpoints/no_other/{0}/{1}/".format(product_name, sim_value)
        model_name = "BLSTM+MLP-aspect_no-other"

    if not exists(modelcheckpoints):
        makedirs(modelcheckpoints)

    checkpoint_path = modelcheckpoints + model_name + ".h5"

    return processed_datasets, test_path, checkpoint_path, sim_value, product_name, product_name_, model_name

def data():
    class My_Custom_Generator(tf.keras.utils.Sequence):
        def __init__(self, aspect, labels, batch_size):
            self.aspect = aspect
            self.labels = labels
            self.batch_size = batch_size

        def __len__(self):
            return (np.ceil(len(self.aspect) / float(self.batch_size))).astype(np.int)

        def __getitem__(self, idx):
            batch_aspc = self.aspect[idx * batch_size : (idx+1) * batch_size]
            batch_y = self.labels[idx * batch_size : (idx+1) * batch_size]

            return ({'aspect_input': batch_aspc}, {'out': batch_y})
    
    
    processed_datasets, test_path, checkpoint_path, sim_value, product_name, _, _ = my_args()
    
    embed_dim = 300
    
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.classes_ = load('{0}{1}-classes.npy'.format(test_path, product_name))
    
    n_classes = len(list(label_encoder.classes_))
    
    print("============= LOAD TRAIN SET ==============")
    train_aspect = load(processed_datasets + 'X_train_aspect.npy')
    train_labels = load(processed_datasets + 'y_train.npy')

    print("aspects: {0}".format(train_aspect.shape))
    print("labels: {0}".format(train_labels.shape))

    print("============= LOAD VALID SET ==============")
    val_aspect = load(processed_datasets + 'X_val_aspect.npy',)
    val_labels = load(processed_datasets + 'y_val.npy')

    print("aspects: {0}".format(val_aspect.shape))
    print("labels: {0}".format(val_labels.shape))
    
    batch_size = 128

    my_training_batch_generator = My_Custom_Generator(train_aspect, 
                                                      train_labels, batch_size)
    my_validation_batch_generator = My_Custom_Generator(val_aspect, 
                                                        val_labels, batch_size)
    
    train_steps = int(len(train_aspect) // batch_size)
    valid_steps =  int(len(val_aspect) // batch_size)
    
    return checkpoint_path, embed_dim, n_classes, batch_size, my_training_batch_generator, my_validation_batch_generator, train_steps, valid_steps


def model(checkpoint_path, embed_dim, n_classes, my_training_batch_generator, my_validation_batch_generator, train_steps, valid_steps):
    
    
    ################### SENTENCE - ASPECT INPUT ###################################
    aspect_embed = Input(shape=(3, embed_dim,), name="aspect_input")
    
    lstm_units = {{choice([50, 100, 150, 200])}}
    
    aspect_forward_layer = LSTM(lstm_units, activation='relu', return_sequences=False)
    aspect_backward_layer = LSTM(lstm_units, activation='relu', return_sequences=False, go_backwards=True)
    aspect_ = Bidirectional(aspect_forward_layer, backward_layer=aspect_backward_layer,
                                 name="BLSTM_aspec")(aspect_embed)
    
    ################### CONCAT AND FULLY CONNECTED ################################
    
    out_ = Dense({{choice([100, 200, 400, 600, 800])}}, activation='relu', name='dense')(aspect_)
    out_ = Dropout(0.5, name="dropout")(out_)
    
    # If we choose 'four', add an additional fourth layer
    fc_number = {{choice(['one','two', 'three'])}}
    dense_2 = {{choice([50, 100, 200, 400, 600])}}
    dense_3 = {{choice([25, 50, 100, 200, 400])}}
    if fc_number == 'two':
        out_ = Dense(dense_2, activation='relu', name='dense1')(out_)
        out_ = Dropout(0.5, name="dropout1")(out_)
    
    elif fc_number == 'three':
        out_ = Dense(dense_2, activation='relu', name='dense1')(out_)
        out_ = Dropout(0.5, name="dropout1")(out_)
        
        out_ = Dense(dense_3, activation='relu', name='dense2')(out_)
        out_ = Dropout(0.5, name="dropout2")(out_)

    out = Dense(n_classes, activation='softmax', name='out')(out_)

    ################### DEFINE AND COMPILE MODEL ##################################
    model = Model(inputs=[aspect_embed], outputs=out)

    model.compile(loss= focal_loss(gamma={{choice([1.0, 2.0, 3.0, 4.0, 5.0])}}, alpha=1.0), 
                  metrics=['acc', AUC(curve='PR', multi_label=False, name='auc')], 
                  optimizer=Adam(0.001))  #   

    model.summary()
    
    """## Fit model"""
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True, save_weights_only=True)
    earlystopper = EarlyStopping(monitor="val_auc", mode="max", patience=5, verbose=1)

    history = model.fit(my_training_batch_generator,
                        steps_per_epoch = train_steps,
                        epochs = 100,
                        verbose = 2,
                        validation_data = my_validation_batch_generator,
                        validation_steps = valid_steps,
                        callbacks=[checkpointer, earlystopper])
    
    model.load_weights(checkpoint_path)
    
    score = model.evaluate(my_validation_batch_generator, verbose=0)

    loss, acc, auc = score

    return {'loss':-auc, 'status': STATUS_OK, 'model': model}


"""# **Run trials**"""

start_time = time.time()
# chose better parameters to model
trials = Trials()
best_run, best_model = optim.minimize(model=model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=20,
                                      trials=trials,
                                      functions=[my_args])

fitting_time = time.time() - start_time
print(">>> FITTING PARAMETERS TIME: {0}".format(fitting_time))

print("Best performing model chosen hyper-parameters:")
print(best_run)


processed_datasets, test_path, checkpoint_path, sim_value, product_name, product_name_, model_name = my_args()

label_encoder = preprocessing.LabelEncoder()
label_encoder.classes_ = load('{0}{1}-classes.npy'.format(test_path, product_name))

target_names_ = label_encoder.classes_
labels_ = label_encoder.transform(label_encoder.classes_)


dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))


best_model_path = "./models/best_model_{0}-{1}_{2}.h5".format(product_name, sim_value, model_name)

best_model.save(best_model_path)


print("============= LOAD VALID SET ==============")
val_aspect = load(processed_datasets + 'X_val_aspect.npy',)
val_labels = load(processed_datasets + 'y_val.npy')

print("aspects: {0}".format(val_aspect.shape))
print("labels: {0}".format(val_labels.shape))


"""# **Evaluation over validation set**"""

y_prob = best_model.predict([val_aspect])
y_pred = np.argmax(y_prob, axis=1)

print("Validation Set - Weighted")
precision_w, recall_w, fscore_w, _ = precision_recall_fscore_support(np.argmax(val_labels, axis=1), y_pred, average='weighted')
print("Precision: {:.4f}\nRecall: {:.4f}\nF-Score: {:.4f}\n".format(precision_w, recall_w, fscore_w))

print("Validation Set - Micro")
precision_mi, recall_mi, fscore_mi, _ = precision_recall_fscore_support(np.argmax(val_labels, axis=1), y_pred, average='micro')
print("Precision: {:.4f}\nRecall: {:.4f}\nF-Score: {:.4f}\n".format(precision_mi, recall_mi, fscore_mi))

print("Validation Set - Macro")
precision_ma, recall_ma, fscore_ma, _ = precision_recall_fscore_support(np.argmax(val_labels, axis=1), y_pred, average='macro')
print("Precision: {:.4f}\nRecall: {:.4f}\nF-Score: {:.4f}\n".format(precision_ma, recall_ma, fscore_ma))


print("=================== VALIDATION CLASSIFICATION REPORT =============================")
print(classification_report(np.argmax(val_labels, axis=1), y_pred, labels=labels_, target_names=target_names_))


print("============= LOAD BESTBUY TEST SET ==============")
bb_test_aspect = load(test_path + 'X_test_bb_aspect.npy')
bb_test_labels = load(test_path + 'y_test_bb.npy')

print("aspects: {0}".format(bb_test_aspect.shape))
print("labels: {0}".format(bb_test_labels.shape))

"""# **Evaluation over bb test set**"""

y_prob_test_bb = best_model.predict([bb_test_aspect])
y_pred_test_bb = np.argmax(y_prob_test_bb, axis=1)

print("Test Set - Weighted")
precision_w, recall_w, fscore_w, _ = precision_recall_fscore_support(np.argmax(bb_test_labels, axis=1), y_pred_test_bb, average='weighted')
print("Precision: {:.4f}\nRecall: {:.4f}\nF-Score: {:.4f}\n".format(precision_w, recall_w, fscore_w))

print("Test Set - Micro")
precision_mi, recall_mi, fscore_mi, _ = precision_recall_fscore_support(np.argmax(bb_test_labels, axis=1), y_pred_test_bb, average='micro')
print("Precision: {:.4f}\nRecall: {:.4f}\nF-Score: {:.4f}\n".format(precision_mi, recall_mi, fscore_mi))

print("Test Set - Macro")
precision_ma, recall_ma, fscore_ma, _ = precision_recall_fscore_support(np.argmax(bb_test_labels, axis=1), y_pred_test_bb, average='macro')
print("Precision: {:.4f}\nRecall: {:.4f}\nF-Score: {:.4f}\n".format(precision_ma, recall_ma, fscore_ma))

print("=================== BB TEST CLASSIFICATION REPORT =============================")
print(classification_report(np.argmax(bb_test_labels, axis=1), y_pred_test_bb, labels=labels_, target_names=target_names_))



print("============= LOAD AMAZON TEST SET ==============")
amzn_test_aspect = load(test_path + 'X_test_amzn_aspect.npy')
amzn_test_labels = load(test_path + 'y_test_amzn.npy')

print("aspects: {0}".format(amzn_test_aspect.shape))
print("labels: {0}".format(amzn_test_labels.shape))


"""# **Evaluation over amzn test set**"""

y_prob_test_amzn = best_model.predict([amzn_test_aspect])
y_pred_test_amzn = np.argmax(y_prob_test_amzn, axis=1)

print("Test Set - Weighted")
precision_w, recall_w, fscore_w, _ = precision_recall_fscore_support(np.argmax(amzn_test_labels, axis=1), y_pred_test_amzn, average='weighted')
print("Precision: {:.4f}\nRecall: {:.4f}\nF-Score: {:.4f}\n".format(precision_w, recall_w, fscore_w))

print("Test Set - Micro")
precision_mi, recall_mi, fscore_mi, _ = precision_recall_fscore_support(np.argmax(amzn_test_labels, axis=1), y_pred_test_amzn, average='micro')
print("Precision: {:.4f}\nRecall: {:.4f}\nF-Score: {:.4f}\n".format(precision_mi, recall_mi, fscore_mi))

print("Test Set - Macro")
precision_ma, recall_ma, fscore_ma, _ = precision_recall_fscore_support(np.argmax(amzn_test_labels, axis=1), y_pred_test_amzn, average='macro')
print("Precision: {:.4f}\nRecall: {:.4f}\nF-Score: {:.4f}\n".format(precision_ma, recall_ma, fscore_ma))

print("=================== AMZN TEST CLASSIFICATION REPORT =============================")
print(classification_report(np.argmax(amzn_test_labels, axis=1), y_pred_test_amzn, labels=labels_, target_names=target_names_))
