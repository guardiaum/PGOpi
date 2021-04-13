#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors, Word2Vec, FastText


# In[ ]:

def split_training_data(label_encoder, df_data):
  X = df_data[['aspect', 'sentence', 'target']]

  Y = []
  for i, row in df_data.iterrows():
    Y.append(row['target'])
  
  Y = label_encoder.transform(Y)
  Y = tf.one_hot(Y, len(label_encoder.classes_)).numpy()

  x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.30, random_state=42, stratify=Y)

  return label_encoder, x_train, x_valid, y_train, y_valid

def split_test_data(label_encoder, test_data):
  test_data['aspectClass'] = test_data['aspectClass'].str.lower()
  test_data.loc[test_data['aspectClass']=='others', 'aspectClass'] = 'other'
  
  x_test = test_data[test_data['aspectClass'].notna() & test_data['aspectClass'].isin(list(label_encoder.classes_))]
  x_test = x_test.rename(columns={'aspectClass':'target'})
  #x_test = x_test[x_test['aspectClass']!='other']   # REMOVES OTHER

  y_test = []
  for i, row in x_test.iterrows():
    y_test.append(row['target'])

  y_test = label_encoder.transform(y_test)
  y_test = tf.one_hot(y_test, len(label_encoder.classes_)).numpy()
  return x_test, y_test

def get_embed_model(path2embed_model):
  model = Word2Vec.load(path2embed_model)
  return model

def pad_embeddings(embedding_model, tokens, MAX_PAD=30, EMB_DIM=100):

    tokens = list(filter(lambda x: x in embedding_model.wv.vocab, tokens))

    if all([True if x == "" else False for x in tokens]):
        padding_embedding = np.array([[0] * EMB_DIM] * MAX_PAD)
        return padding_embedding
    elif len(tokens) < MAX_PAD:
        embedding = embedding_model.wv[tokens]
        padding_embedding = np.pad(embedding, ((0, MAX_PAD - len(tokens)), (0, 0)), 'constant')
        return padding_embedding
    else:
        embedding = embedding_model.wv[tokens[0:MAX_PAD]]
        return embedding
    
def get_char_dict():
    # dictionary of all possible characters

    char2Idx = {'UNK': 0}
    for c in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|<>":
      char2Idx[c] = len(char2Idx)

    return char2Idx

def pad_tokens(tokens, char2Idx, WORD_MAX_PAD=12):
    for j, token in enumerate(tokens):
        chars = [c for c in token]  # data[0] is the character

        chars = [char2Idx[c] if c in char2Idx.keys() else char2Idx['UNK'] for c in chars]

        if len(chars) < WORD_MAX_PAD:
          return np.pad(chars, (0, WORD_MAX_PAD - len(chars)), 'constant')
        else:
          return np.array(chars[:WORD_MAX_PAD])

def pad_embed_tkns(tokenized_sent, char2Idx, SENT_MAX_PAD=3):

    if len(tokenized_sent) < SENT_MAX_PAD:
      tokenized_sent = np.array([np.pad(tokenized_sent, (0, SENT_MAX_PAD - len(tokenized_sent)), 'constant')])
    else:
      tokenized_sent = np.array([tokenized_sent[:SENT_MAX_PAD]])
    
    tokenized_sent = np.apply_along_axis(pad_tokens, 0, tokenized_sent, char2Idx)
    
    return tokenized_sent.T

def focal_loss(gamma=2.0, alpha=4.0):
    gamma = float(gamma)
    alpha = float(alpha)
    
    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002
        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]
        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})
        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed

class My_Custom_Generator(tf.keras.utils.Sequence):
    def __init__(self, sentence, aspect, aspc_tkns, labels, batch_size):
        self.sentence = sentence
        self.aspect = aspect
        self.aspc_tkns = aspc_tkns
        self.labels = labels
        self.batch_size = batch_size
        
    def __len__(self):
        return (np.ceil(len(self.sentence) / float(self.batch_size))).astype(np.int)
    
    def __getitem__(self, idx):
        batch_sent = self.sentence[idx * batch_size : (idx+1) * self.batch_size]
        batch_aspc = self.aspect[idx * batch_size : (idx+1) * self.batch_size]
        batch_aspc_tkns = self.aspc_tkns[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        
        return ({'sentence_input': batch_sent, 'aspect_input': batch_aspc, 
                'aspc_tkns_input': batch_aspc_tkns}, {'out': batch_y})