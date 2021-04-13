#!/usr/bin/env python
# coding: utf-8


from os import listdir, makedirs
from os.path import isfile, join, exists
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors, Word2Vec, FastText
from util import *
from numpy import load
import numpy as np
import pandas as pd
import time
import nltk
nltk.download('punkt')


with_other = True   # True/False
sim_value = "9"
product_name_ = "cel"  # cel, cam, dvd, rou, lap
product_name = "cells"   # cells, cameras, dvds, routers, laptops

root = "./"
datasets_dir = root + "datasets/"
train_dir = datasets_dir + "all_train_sets/"
path_to_word_embeds_models = datasets_dir + "w2v/"

processed_datasets, label_encoder_path = '', ''
if with_other==True:
    processed_datasets = "./datasets/data_preprocessed/with_other/{0}/{1}/".format(product_name, sim_value)
    label_encoder_path = "./datasets/data_preprocessed/with_other/{0}/test_sets/".format(product_name)
else:
    processed_datasets = "./datasets/data_preprocessed/no_other/{0}/{1}/".format(product_name, sim_value)
    label_encoder_path = "./datasets/data_preprocessed/no_other/{0}/test_sets/".format(product_name)

if not exists(processed_datasets):
    makedirs(processed_datasets)

start_time = time.time()

# load training and test data with computed similarities
train_data = pd.read_csv(train_dir + product_name_ + "-train-"+ sim_value +".tsv", sep="\t", names=["target", "aspect", "sentiment", "sentence"])


label_encoder = preprocessing.LabelEncoder()
label_encoder.classes_ = load('{0}{1}-classes.npy'.format(label_encoder_path, product_name))
    

'''read label encode'''
targets_to_keep = list(label_encoder.classes_)

train_data = train_data[train_data['target'].isin(targets_to_keep)]


# remove other if needed
if with_other == False:
    train_data = train_data[train_data['target'] != 'other']


train_data['target'].unique()


# split training set into train and validation sets. Split samples and labels (also encode labels to hotencode categorical format)
label_encoder, x_train, x_valid, y_train, y_valid = split_training_data(label_encoder, train_data)

del train_data

x_train['target'].unique()
x_train['target'].nunique()
x_valid['target'].nunique()


# Load pretrained word-embedding model
embed_model = get_embed_model(path_to_word_embeds_models + "modelo-" + product_name + "-pgopi.bin")

# # Train and Dev Sets

n_classes = len(list(label_encoder.classes_))

embed_dim = 300

tokenizer = nltk.RegexpTokenizer(r"\w+")

char2Idx = get_char_dict()

print("============= TRAINING SET ==============")

train_sentences_padded, train_aspect_padded = [], []
for i, row in tqdm(x_train.iterrows()):
    train_sentences_padded.append(pad_embeddings(embed_model, tokenizer.tokenize(str(row['sentence']).lower()), EMB_DIM=embed_dim))

train_sentences = np.array(train_sentences_padded)
del train_sentences_padded
print("sentences: {0}".format(train_sentences.shape))
np.save(processed_datasets + 'X_train_sentences.npy', train_sentences)
del train_sentences

for i, row in tqdm(x_train.iterrows()):
    train_aspect_padded.append(pad_embeddings(embed_model, tokenizer.tokenize(row['aspect'].lower()), MAX_PAD=3, EMB_DIM=embed_dim))

train_aspect = np.array(train_aspect_padded)
del train_aspect_padded
print("aspects: {0}".format(train_aspect.shape))
np.save(processed_datasets + 'X_train_aspect.npy', train_aspect)
del train_aspect

del x_train

train_labels = np.array(y_train)
del y_train
print("labels: {0}".format(train_labels.shape))
np.save(processed_datasets + 'y_train.npy', train_labels)
del train_labels

print("============= VALID SET ==============")

val_sentences_padded, val_aspect_padded = [], []
for i, row in tqdm(x_valid.iterrows()):
    val_sentences_padded.append(pad_embeddings(embed_model, tokenizer.tokenize(row['sentence'].lower()), EMB_DIM=embed_dim))
    
val_sentences = np.array(val_sentences_padded)    
del val_sentences_padded 
print("sentences: {0}".format(val_sentences.shape))
np.save(processed_datasets + 'X_val_sentences.npy', val_sentences)
del val_sentences

for i, row in tqdm(x_valid.iterrows()):
    val_aspect_padded.append(pad_embeddings(embed_model, tokenizer.tokenize(row['aspect'].lower()), MAX_PAD=3, EMB_DIM=embed_dim))

val_aspect = np.array(val_aspect_padded)
del val_aspect_padded
print("aspects: {0}".format(val_aspect.shape))
np.save(processed_datasets + 'X_val_aspect.npy', val_aspect)
del val_aspect

del x_valid

val_labels = np.array(y_valid)
del y_valid
print("labels: {0}".format(val_labels.shape))
np.save(processed_datasets + 'y_val.npy', val_labels)
del val_labels

print("EXECUTION TIME: %s seconds" % (time.time() - start_time))
