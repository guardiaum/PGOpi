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
import csv
from csv import DictWriter

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


def append_dict_as_row(file_name, dict_of_elem, field_names):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        dict_writer = DictWriter(write_obj, fieldnames=field_names)
        # Add dictionary as wor in the csv
        dict_writer.writerow(dict_of_elem)

class My_Custom_Generator(tf.keras.utils.Sequence):
    def __init__(self, sentence, aspect, labels, batch_size):
        self.sentence = sentence
        self.aspect = aspect
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.sentence) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_sent = self.sentence[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_aspc = self.aspect[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

        return ({'sentence_input': batch_sent, 'aspect_input': batch_aspc}, {'out': batch_y})


# In[2]:


with_other = True
sim_value = "5"
product_name = "cells"   # cells, cameras, dvds, routers, laptops
product_name_ = "cel"  # cel, cam, dvd, rou, lap

repetitions = 10

# CHANGE IT
params_id_select = {'Dense': 0, 'dense_2': 3, 'dense_3': 3, 'fc_number': 0, 'gamma': 1, 'lstm_units': 0}


# DO NOT CHANGE!
params_values = {'lstm_units': [50, 100, 150, 200],
                 'Dense': [100, 200, 400, 600, 800],
                 'fc_number': ['one','two', 'three'],
                 'dense_2': [50, 100, 200, 400, 600],
                 'dense_3': [25, 50, 100, 200, 400],
                 'gamma': [1.0, 2.0, 3.0, 4.0, 5.0]}

params_chosen = {}
for key in params_values:
    params_chosen[key] = params_values[key][params_id_select[key]]

root = "./"
datasets_dir = root + "datasets/"

model_name, processed_datasets, label_encoder_path, modelcheckpoints = '', '', '', ''
if with_other==True:
    processed_datasets = datasets_dir + "data_preprocessed/with_other/{0}/{1}/".format(product_name, sim_value)
    test_path = "./datasets/data_preprocessed/with_other/{0}/test_sets/".format(product_name)
    modelcheckpoints = root + "models/checkpoints/with_other/{0}/{1}/".format(product_name, sim_value)
    model_name = "BLSTM+co+MLP_with-other"
    output_filename = "{0}exp_{1}_{2}_with-other_{3}.csv".format("./experiments/", product_name, sim_value, model_name)
else:
    processed_datasets = datasets_dir + "data_preprocessed/no_other/{0}/{1}/".format(product_name, sim_value)
    test_path = "./datasets/data_preprocessed/no_other/{0}/test_sets/".format(product_name)
    modelcheckpoints = root + "models/checkpoints/no_other/{0}/{1}/".format(product_name, sim_value)
    model_name = "BLSTM+co+MLP_no-other"
    output_filename = "{0}exp_{1}_{2}_no-other_{3}.csv".format("./experiments/", product_name, sim_value, model_name)

if not exists(modelcheckpoints):
    makedirs(modelcheckpoints)

field_names = ["iter", 'val_loss', 'val_acc', "val_auc", 
               "bestbuy_precision_w", "bestbuy_recall_w", "bestbuy_fscore_w",
               "bestbuy_precision_mi", "bestbuy_recall_mi", "bestbuy_fscore_mi",
               "bestbuy_precision_ma", "bestbuy_recall_ma", "bestbuy_fscore_ma",
               "amazon_precision_w", "amazon_recall_w", "amazon_fscore_w",
               "amazon_precision_mi", "amazon_recall_mi", "amazon_fscore_mi",
               "amazon_precision_ma", "amazon_recall_ma", "amazon_fscore_ma", "time_spent"]

with open(output_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(field_names)
    
checkpoint_path = modelcheckpoints + model_name + ".h5"

embed_dim = 300

label_encoder = preprocessing.LabelEncoder()
label_encoder.classes_ = load('{0}{1}-classes.npy'.format(test_path, product_name))

n_classes = len(list(label_encoder.classes_))

print("============= LOAD TRAIN SET ==============")
train_sentences = load(processed_datasets + 'X_train_sentences.npy')
train_aspect = load(processed_datasets + 'X_train_aspect.npy')
train_labels = load(processed_datasets + 'y_train.npy')

print("sentences: {0}".format(train_sentences.shape))
print("aspects: {0}".format(train_aspect.shape))
print("labels: {0}".format(train_labels.shape))

print("============= LOAD VALID SET ==============")
val_sentences = load(processed_datasets + 'X_val_sentences.npy')
val_aspect = load(processed_datasets + 'X_val_aspect.npy',)
val_labels = load(processed_datasets + 'y_val.npy')

print("sentences: {0}".format(val_sentences.shape))
print("aspects: {0}".format(val_aspect.shape))
print("labels: {0}".format(val_labels.shape))

batch_size = 128

my_training_batch_generator = My_Custom_Generator(train_sentences, train_aspect, 
                                                  train_labels, batch_size)
my_validation_batch_generator = My_Custom_Generator(val_sentences, val_aspect, 
                                                    val_labels, batch_size)

train_steps = int(len(train_sentences) // batch_size)
valid_steps =  int(len(val_sentences) // batch_size)


# In[ ]:

total_time = 0

for iteration in range(1, repetitions + 1):
    start_time = time.time()
    
    print("START ITERATION {0}/{1}".format(iteration, repetitions))

    ################### SENTENCE - ASPECT INPUT ###################################
    sentence_embed = Input(shape=(30, embed_dim,), name="sentence_input")
    aspect_embed = Input(shape=(3, embed_dim,), name="aspect_input")
    
    lstm_units = params_chosen['lstm_units']
    
    sentence_forward_layer = LSTM(lstm_units, activation='relu', return_sequences=False)
    sentence_backward_layer = LSTM(lstm_units, activation='relu', return_sequences=False, go_backwards=True)
    sentence_ = Bidirectional(sentence_forward_layer, backward_layer=sentence_backward_layer, 
                              name="BLSTM_sent")(sentence_embed)

    aspect_forward_layer = LSTM(lstm_units, activation='relu', return_sequences=False)
    aspect_backward_layer = LSTM(lstm_units, activation='relu', return_sequences=False, go_backwards=True)
    aspect_ = Bidirectional(aspect_forward_layer, backward_layer=aspect_backward_layer,
                                 name="BLSTM_aspec")(aspect_embed)
    
    co_1 = tf.keras.layers.Attention(name='co-attention1')([sentence_, aspect_])
    
    co_2 = tf.keras.layers.Attention(name='co-attention2')([aspect_, sentence_])
    
    ################### CONCAT AND FULLY CONNECTED ################################
    concat_ = Concatenate()([co_1, co_2])
    
    out_ = Dense(params_chosen['Dense'], activation='relu', name='dense')(concat_)
    out_ = Dropout(0.5, name="dropout")(out_)
    
    # If we choose 'four', add an additional fourth layer
    fc_number = params_chosen['fc_number']
    dense_2 = params_chosen['dense_2']
    dense_3 = params_chosen['dense_3']
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
    model = Model(inputs=[sentence_embed, aspect_embed], outputs=out)

    model.compile(loss= focal_loss(gamma=params_chosen['gamma'], alpha=1.0), 
                  metrics=['acc', AUC(curve='PR', multi_label=False, name='auc')], 
                  optimizer=Adam(0.001))  

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
    
    time_spent = time.time() - start_time
    
    model.load_weights(checkpoint_path)

    score = model.evaluate(my_validation_batch_generator, verbose=0)

    loss, val_acc, val_auc = score
    
    print("[VALIDATION SET] loss: {0}, val_acc: {1}, val_auc: {2}".format(loss, val_acc, val_auc))
    
    ### SAVE BEST MODEL WEIGHTS TO FILE
    best_models_dir = "./models/experiment/{0}/".format(product_name)
    
    if not exists(best_models_dir):
        makedirs(best_models_dir)
                  
    best_model_path = "{0}best_model_{1}_{2}.h5".format(best_models_dir, model_name, iteration)

    model.save(best_model_path)
                  
    print("============= LOAD TEST SET [BESTBUY] ==============")
    test_sentences = load(test_path + 'X_test_bb_sentences.npy')
    test_aspect = load(test_path + 'X_test_bb_aspect.npy')
    test_labels = load(test_path + 'y_test_bb.npy')

    print("[BESTBUY] sentences: {0}".format(test_sentences.shape))
    print("[BESTBUY] aspects: {0}".format(test_aspect.shape))
    print("[BESTBUY] labels: {0}".format(test_labels.shape))
                  
    """# **Evaluation over test set**"""

    y_prob_test = model.predict([test_sentences, test_aspect])
    y_pred_test = np.argmax(y_prob_test, axis=1)

    print("[BESTBUY] Test Set - Weighted")
    precision_w, recall_w, fscore_w, _ = precision_recall_fscore_support(np.argmax(test_labels, axis=1), y_pred_test, average='weighted')
    print("[BESTBUY] Precision: {:.4f}\n[BESTBUY] Recall: {:.4f}\n[BESTBUY] F-Score: {:.4f}\n".format(precision_w, recall_w, fscore_w))

    print("[BESTBUY] Test Set - Micro")
    precision_mi, recall_mi, fscore_mi, _ = precision_recall_fscore_support(np.argmax(test_labels, axis=1), y_pred_test, average='micro')
    print("[BESTBUY] Precision: {:.4f}\n[BESTBUY] Recall: {:.4f}\n[BESTBUY] F-Score: {:.4f}\n".format(precision_mi, recall_mi, fscore_mi))

    print("[BESTBUY] Test Set - Macro")
    precision_ma, recall_ma, fscore_ma, _ = precision_recall_fscore_support(np.argmax(test_labels, axis=1), y_pred_test, average='macro')
    print("[BESTBUY] Precision: {:.4f}\n[BESTBUY] Recall: {:.4f}\n[BESTBUY] F-Score: {:.4f}\n".format(precision_ma, recall_ma, fscore_ma))
                  
    print("============= LOAD TEST SET [AMAZON] ==============")
                  
    test_sentences_amzn = load(test_path + 'X_test_amzn_sentences.npy')
    test_aspect_amzn = load(test_path + 'X_test_amzn_aspect.npy')
    test_labels_amzn = load(test_path + 'y_test_amzn.npy')

    print("[AMAZON] sentences: {0}".format(test_sentences_amzn.shape))
    print("[AMAZON] aspects: {0}".format(test_aspect_amzn.shape))
    print("[AMAZON] labels: {0}".format(test_labels_amzn.shape))
                  
    """# **Evaluation over test set**"""

    y_prob_test_amzn = model.predict([test_sentences_amzn, test_aspect_amzn])
    y_pred_test_amzn = np.argmax(y_prob_test_amzn, axis=1)

    print("[AMAZON] Test Set - Weighted")
    precision_w_amzn, recall_w_amzn, fscore_w_amzn, _ = precision_recall_fscore_support(np.argmax(test_labels_amzn, axis=1), y_pred_test_amzn, average='weighted')
    print("[AMAZON] Precision: {:.4f}\n[AMAZON] Recall: {:.4f}\n[AMAZON] F-Score: {:.4f}\n".format(precision_w_amzn, recall_w_amzn, fscore_w_amzn))

    print("[AMAZON] Test Set - Micro")
    precision_mi_amzn, recall_mi_amzn, fscore_mi_amzn, _ = precision_recall_fscore_support(np.argmax(test_labels_amzn, axis=1), y_pred_test_amzn, average='micro')
    print("[AMAZON] Precision: {:.4f}\n[AMAZON] Recall: {:.4f}\n[AMAZON] F-Score: {:.4f}\n".format(precision_mi_amzn, recall_mi_amzn, fscore_mi_amzn))

    print("[AMAZON] Test Set - Macro")
    precision_ma_amzn, recall_ma_amzn, fscore_ma_amzn, _ = precision_recall_fscore_support(np.argmax(test_labels_amzn, axis=1), y_pred_test_amzn, average='macro')
    print("[AMAZON] Precision: {:.4f}\n[AMAZON] Recall: {:.4f}\n[AMAZON] F-Score: {:.4f}\n".format(precision_ma_amzn, recall_ma_amzn, fscore_ma_amzn))
    
    row_dict = {"iter": iteration, 'val_loss': loss, 'val_acc': val_acc, "val_auc": val_auc,
                "bestbuy_precision_w": precision_w, "bestbuy_recall_w": recall_w, "bestbuy_fscore_w": fscore_w,
                "bestbuy_precision_mi": precision_mi, "bestbuy_recall_mi": recall_mi, "bestbuy_fscore_mi": fscore_mi,
                "bestbuy_precision_ma": precision_ma, "bestbuy_recall_ma": recall_ma, "bestbuy_fscore_ma": fscore_ma,
                "amazon_precision_w": precision_w_amzn, "amazon_recall_w": recall_w_amzn, "amazon_fscore_w": fscore_w_amzn,
                "amazon_precision_mi": precision_mi_amzn, "amazon_recall_mi": recall_mi_amzn, "amazon_fscore_mi": fscore_mi_amzn,
                "amazon_precision_ma": precision_ma_amzn, "amazon_recall_ma": recall_ma_amzn, "amazon_fscore_ma": fscore_ma_amzn, "time_spent":time_spent}
    
    total_time = total_time + time_spent
    # WRITE RESULTS TO FILE
    append_dict_as_row(output_filename, row_dict, field_names)
                  
print("TOTAL EXECUTION TIME: {0} SECONDS".format(total_time))
