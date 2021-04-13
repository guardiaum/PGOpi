#!/usr/bin/env python
# coding: utf-8


from os import listdir, makedirs
from os.path import isfile, join, exists
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors, Word2Vec, FastText
from util import *
import numpy as np
import pandas as pd
import time
import nltk
nltk.download('punkt')


with_other = True   # True/False


root = "./"
datasets_dir = root + "datasets/"
test_dir = datasets_dir + "test_sets/"
path_to_word_embeds_models = datasets_dir + "w2v/"

products_name_ = ["cel", "cam", "dvd", "rou", "lap"]
products_name = ["cells", "cameras", "dvds", "routers", "laptops"]

running_time = 0

for product_name in products_name:
    
    print("+++++++++++ {0} ++++++++++++++++++++++++++++++++++++++++++++++++".format(product_name.upper()))
    processed_datasets = ''
    if with_other==True:
        processed_datasets = "./datasets/data_preprocessed/with_other/{0}/test_sets/".format(product_name)
    else:
        processed_datasets = "./datasets/data_preprocessed/no_other/{0}/test_sets/".format(product_name)

    if not exists(processed_datasets):
        makedirs(processed_datasets)


    start_time = time.time()

    bestbuy_test_data = pd.read_csv(test_dir + "test-" + product_name + "-bestbuy.csv" , index_col=0)
    amazon_test_data = pd.read_csv(test_dir + "test-" + product_name + "-amazon.csv")

    # changes aspectClass in bestbuy test set to use the label 'other' instead of 'others'
    bestbuy_test_data['aspectClass'] = bestbuy_test_data['aspectClass'].str.lower()
    bestbuy_test_data.loc[bestbuy_test_data['aspectClass']=='others', 'aspectClass'] = 'other'

    # preprocesses amazon test set to match columns and labels in the other sets
    amazon_test_data = amazon_test_data.rename({'attribute':'aspectClass'}, axis='columns')
    amazon_test_data['aspectClass'] = amazon_test_data['aspectClass'].str.lower()
    amazon_test_data.loc[amazon_test_data['aspectClass']=='others', 'aspectClass'] = 'other'

    # remove other
    if with_other == False:
        train_data = train_data[train_data['target'] != 'other']
        bestbuy_test_data = bestbuy_test_data[bestbuy_test_data['aspectClass'] != 'other']
        amazon_test_data = amazon_test_data[amazon_test_data['aspectClass'] != 'other']
    
    
    # Unify the list of targets removing 'nan' targets
    bb = bestbuy_test_data['aspectClass'].unique().tolist()
    amz = amazon_test_data['aspectClass'].unique().tolist()

    merge = set(bb + amz)

    targets_to_keep = [x for x in merge if str(x) != 'nan']
    
    print(">> Targets to keep: ", targets_to_keep)
    
    bestbuy_test_data = bestbuy_test_data[bestbuy_test_data['aspectClass'].isin(targets_to_keep)]                                        
    amazon_test_data = amazon_test_data[amazon_test_data['aspectClass'].isin(targets_to_keep)]                                         
    
    # fit label encoder
    le = preprocessing.LabelEncoder()
    le.fit(targets_to_keep)
    
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)
    
    # preprocess bestbuy test set to split samples and labels (also encode labels to hotencode categorical format)
    x_bestbuy_test, y_bestbuy_test = split_test_data(le, bestbuy_test_data)
    
    # preprocess test set to split samples and labels (also encode labels to hotencode categorical format)
    x_amazon_test, y_amazon_test = split_test_data(le, amazon_test_data)                                        
    
    
    del bestbuy_test_data
    del amazon_test_data                                        

    print("> bestbuy targets ", x_bestbuy_test['target'].unique().tolist())
    print("# bestbuy targets ", x_bestbuy_test['target'].nunique())
    print("> amazon targets ", x_amazon_test['target'].unique().tolist())
    print("# amazon targets ", x_bestbuy_test['target'].nunique())


    # Save label encoder to file
    np.save('{0}{1}-classes.npy'.format(processed_datasets, product_name), le.classes_)                                                                              

    # Load pretrained word-embedding model
    embed_model = get_embed_model(path_to_word_embeds_models + "modelo-" + product_name + "-pgopi.bin")


    # # Train and Dev Sets

    n_classes = len(le.classes_)

    embed_dim = 300

    tokenizer = nltk.RegexpTokenizer(r"\w+")

    # # BestBuy Testset

    test_sentences_padded, test_aspect_padded = [], []
    for i, row in tqdm(x_bestbuy_test.iterrows()):
        test_sentences_padded.append(pad_embeddings(embed_model, tokenizer.tokenize(row['sentence'].lower()), EMB_DIM=embed_dim))
        test_aspect_padded.append(pad_embeddings(embed_model, tokenizer.tokenize(row['aspect'].lower()), MAX_PAD=3, EMB_DIM=embed_dim))

    del x_bestbuy_test

    bb_test_sentences = np.array(test_sentences_padded)
    bb_test_aspect = np.array(test_aspect_padded)
    bb_test_labels = np.array(y_bestbuy_test)

    del test_sentences_padded
    del test_aspect_padded
    del y_bestbuy_test

    print("============= BESTBUY TEST SET ==============")
    print("sentences: {0}".format(bb_test_sentences.shape))
    print("aspects: {0}".format(bb_test_aspect.shape))
    print("labels: {0}".format(bb_test_labels.shape))

    np.save(processed_datasets + 'X_test_bb_sentences.npy', bb_test_sentences)
    np.save(processed_datasets + 'X_test_bb_aspect.npy', bb_test_aspect)
    np.save(processed_datasets + 'y_test_bb.npy', bb_test_labels)


    del bb_test_sentences
    del bb_test_aspect
    del bb_test_labels


    x_amazon_test.head()


    # # Amazon Testset

    test_sentences_padded, test_aspect_padded = [], []
    for i, row in tqdm(x_amazon_test.iterrows()):
        test_sentences_padded.append(pad_embeddings(embed_model, tokenizer.tokenize(row['sentence'].lower()), EMB_DIM=embed_dim))
        test_aspect_padded.append(pad_embeddings(embed_model, tokenizer.tokenize(row['aspect'].lower()), MAX_PAD=3, EMB_DIM=embed_dim))

    del x_amazon_test

    amzn_test_sentences = np.array(test_sentences_padded)
    amzn_test_aspect = np.array(test_aspect_padded)
    amzn_test_labels = np.array(y_amazon_test)

    del test_sentences_padded
    del test_aspect_padded
    del y_amazon_test

    del embed_model

    print("============= AMAZON TEST SET ==============")
    print("sentences: {0}".format(amzn_test_sentences.shape))
    print("aspects: {0}".format(amzn_test_aspect.shape))
    print("labels: {0}".format(amzn_test_labels.shape))

    np.save(processed_datasets + 'X_test_amzn_sentences.npy', amzn_test_sentences)
    np.save(processed_datasets + 'X_test_amzn_aspect.npy', amzn_test_aspect)
    np.save(processed_datasets + 'y_test_amzn.npy', amzn_test_labels)
    
    prod_time = time.time() - start_time
    print("EXECUTION TIME: %s seconds" % (prod_time))
    running_time = running_time + prod_time

print("Total time: {0} seconds".format(running_time))