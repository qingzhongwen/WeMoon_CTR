# -*- coding:utf-8 -*-

from collections import OrderedDict, namedtuple, defaultdict
from itertools import chain
from copy import copy
import tensorflow as tf
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import Embedding, Input, Flatten
from tensorflow.python.keras.regularizers import l2
from DeepRecall.tools.features_typy_dict import SparseFeat, VarLenSparseFeat, DenseFeat
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras import backend as K
from DeepRecall.layers.utils import Hash, concat_func
from DeepRecall.layers.sequence import WeightedSequenceLayer, SequencePoolingLayer


def build_input_features(feature_columns, prefix=''):
    input_features = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            input_features[fc.name] = Input(
                shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, DenseFeat):
            input_features[fc.name] = Input(
                shape=(fc.dimension,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, VarLenSparseFeat):
            input_features[fc.name] = Input(shape=(fc.maxlen,), name=prefix + fc.name,
                                            dtype=fc.dtype)
            if fc.weight_name is not None:
                input_features[fc.weight_name] = Input(shape=(fc.maxlen, 1), name=prefix + fc.weight_name,
                                                       dtype="float32")
            if fc.length_name is not None:
                input_features[fc.length_name] = Input((1,), name=prefix + fc.length_name, dtype='int32')

        else:
            raise TypeError("Invalid feature column type,got", type(fc))

    return input_features


def create_embedding_matrix(feature_columns, l2_reg, init_std, seed, prefix="", seq_mask_zero=True):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, init_std, seed,
                                            l2_reg, prefix=prefix + 'sparse', seq_mask_zero=seq_mask_zero)
    return sparse_emb_dict

def create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, init_std, seed, l2_reg,
                          prefix='sparse_', seq_mask_zero=True):
    sparse_embedding = {feat.embedding_name: Embedding(feat.vocabulary_size, feat.embedding_dim,
                                                       embeddings_initializer=RandomNormal(
                                                           mean=0.0, stddev=init_std, seed=seed),
                                                       embeddings_regularizer=l2(l2_reg),
                                                       name=prefix + '_emb_' + feat.embedding_name) for feat in
                        sparse_feature_columns}

    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            # if feat.name not in sparse_embedding:
            sparse_embedding[feat.embedding_name] = Embedding(feat.vocabulary_size, feat.embedding_dim,
                                                              embeddings_initializer=RandomNormal(
                                                                  mean=0.0, stddev=init_std, seed=seed),
                                                              embeddings_regularizer=l2(
                                                                  l2_reg),
                                                              name=prefix + '_seq_emb_' + feat.name,
                                                              mask_zero=seq_mask_zero)
    return sparse_embedding

def input_from_feature_columns(features, feature_columns, l2_reg, init_std, seed, embedding_matrix_dict,
                               prefix='', seq_mask_zero=True, support_dense=True, support_group=False):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

    # embedding_matrix_dict = create_embedding_matrix(feature_columns, l2_reg, init_std, seed, prefix=prefix,
    #                                                 seq_mask_zero=seq_mask_zero)
    group_sparse_embedding_dict = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns)
    dense_value_list = get_dense_input(features, feature_columns)
    if not support_dense and len(dense_value_list) > 0:
        raise ValueError("DenseFeat is not supported in dnn_feature_columns")

    sequence_embed_dict = varlen_embedding_lookup(embedding_matrix_dict, features, varlen_sparse_feature_columns)
    group_varlen_sparse_embedding_dict = get_varlen_pooling_list(sequence_embed_dict, features,
                                                                 varlen_sparse_feature_columns)
    group_embedding_dict = mergeDict(group_sparse_embedding_dict, group_varlen_sparse_embedding_dict)
    if not support_group:
        group_embedding_dict = list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict, dense_value_list


def embedding_lookup(sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, return_feat_list=(),
                     mask_feat_list=(), to_list=False):
    group_embedding_dict = defaultdict(list)
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if (len(return_feat_list) == 0 or feature_name in return_feat_list):
            if fc.use_hash:
                lookup_idx = Hash(fc.vocabulary_size, mask_zero=(feature_name in mask_feat_list))(
                    sparse_input_dict[feature_name])
            else:
                lookup_idx = sparse_input_dict[feature_name]

            group_embedding_dict[fc.group_name].append(sparse_embedding_dict[embedding_name](lookup_idx))
    if to_list:
        return list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict


def varlen_embedding_lookup(embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            lookup_idx = Hash(fc.vocabulary_size, mask_zero=True)(sequence_input_dict[feature_name])
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)
    return varlen_embedding_vec_dict


def get_varlen_pooling_list(embedding_dict, features, varlen_sparse_feature_columns, to_list=False):
    pooling_vec_list = defaultdict(list)
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        combiner = fc.combiner
        feature_length_name = fc.length_name
        if feature_length_name is not None:
            if fc.weight_name is not None:
                seq_input = WeightedSequenceLayer(weight_normalization=fc.weight_norm)(
                    [embedding_dict[feature_name], features[feature_length_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=False)(
                [seq_input, features[feature_length_name]])
        else:
            if fc.weight_name is not None:
                seq_input = WeightedSequenceLayer(weight_normalization=fc.weight_norm, supports_masking=True)(
                    [embedding_dict[feature_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=True)(
                seq_input)
        pooling_vec_list[fc.group_name].append(vec)
    if to_list:
        return chain.from_iterable(pooling_vec_list.values())
    return pooling_vec_list

def get_dense_input(features, feature_columns):
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        dense_input_list.append(features[fc.name])
    return dense_input_list


def dict_to_list(inputs_dict):
    return list(inputs_dict.values())

def get_linear_inputs(numeric_inputs_dict,linear_cat_embeddings_dict):
    numeric_inputs_list = dict_to_list(numeric_inputs_dict)
    linear_cat_embeddings_list = dict_to_list(linear_cat_embeddings_dict)
    if len(numeric_inputs_list) > 0 and len(linear_cat_embeddings_list) > 0:
        # all of the linear_cat_embeddings categorical features accumulate  shape：(None, 1, 1) --> (None, 1, 7)
        categorical_input = concat_func(linear_cat_embeddings_list)
        # shape (None, 2)
        numeric_input = concat_func(numeric_inputs_list)
    else:
        raise NotImplementedError
    return [categorical_input, numeric_input]


def combined_dnn_input(categorical_embedding_list, numeric_value_list):
    if len(categorical_embedding_list) > 0 and len(numeric_value_list) > 0:
        # 7 * (None, 1, 8) = (None, 1, 56) Flatten --> (None, 56)
        categorical_dnn_input = Flatten()(concat_func(categorical_embedding_list))
        # 2 * (None, 1) = (None, 2) Flatten --> (None, 2)
        numeric_dnn_input = Flatten()(concat_func(numeric_value_list))
        # (None, 58)
        return concat_func([categorical_dnn_input, numeric_dnn_input])
    elif len(categorical_embedding_list) > 0:
        return Flatten()(concat_func(categorical_embedding_list))
    elif len(numeric_value_list) > 0:
        return Flatten()(concat_func(numeric_value_list))
    else:
        raise NotImplementedError

def concat_func(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)

def reduce_mean(input_tensor,
               axis=None,
               keep_dims=False,
               name=None):
    return  tf.reduce_mean(input_tensor,
               axis=axis,
               keepdims=keep_dims,
               name=name)


def reduce_sum(input_tensor,
               axis=None,
               keep_dims=False,
               name=None):
    return tf.reduce_sum(input_tensor,
               axis=axis,
               keepdims=keep_dims,
               name=name)

def reduce_max(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    if tf.__version__ < '2.0.0':
        return tf.reduce_max(input_tensor,
                   axis=axis,
                   keep_dims=keep_dims,
                   name=name,
                   reduction_indices=reduction_indices)
    else:
        return  tf.reduce_max(input_tensor,
                   axis=axis,
                   keepdims=keep_dims,
                   name=name)

def div(x, y, name=None):
    return tf.divide(x, y, name=name)

def softmax(logits, dim=-1, name=None):
    return tf.nn.softmax(logits, axis=dim, name=name)

class Add(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Add, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs,list):
            return inputs
        if len(inputs) == 1:
            return inputs[0]
        if len(inputs) == 0:
            return tf.constant([[0.0]])
        return tf.keras.layers.add(inputs)

def add_func(inputs):
    return Add()(inputs)

def mergeDict(a, b):
    c = defaultdict(list)
    for k, v in a.items():
        c[k].extend(v)
    for k, v in b.items():
        c[k].extend(v)
    return c

def recall_N(y_true, y_pred, N=50):
    return len(set(y_pred[:N]) & set(y_true)) * 1.0 / len(y_true)


def sampledsoftmaxloss(y_true, y_pred):
    return K.mean(y_pred)

def get_item_embedding(item_embedding, item_input_layer):
    return Lambda(lambda x: tf.squeeze(tf.gather(item_embedding, x), axis=1))(
        item_input_layer)

