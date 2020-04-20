# -*- coding:utf-8 -*-

"""
Based on TensorFlow2.0
Author:
    Qingzhongwen, VX: ziqingxiansheng
"""


from collections import OrderedDict
import tensorflow as tf
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import Embedding, Input, Flatten
from tensorflow.python.keras.regularizers import l2
from deep_models.keras_models.layers.features_typy_dict import Categorical_Feature, Numeric_Feat


def get_feature_inputs_dict(all_feat_cols_dict_list):
    num_feats_input_dict = OrderedDict()
    cat_feats_input_dict = OrderedDict()
    for fc in all_feat_cols_dict_list:
        if isinstance(fc, Categorical_Feature):
            cat_feats_input_dict[fc.name] = Input(shape=(fc.dimension,), name=fc.name, dtype=fc.dtype)
        elif isinstance(fc, Numeric_Feat):
            num_feats_input_dict[fc.name] = Input(shape=(fc.dimension,), name=fc.name, dtype=fc.dtype)
        else:
            raise TypeError("Please input valid feature column type", type(fc))
    return num_feats_input_dict,cat_feats_input_dict

def get_cat_embeddings(feat_columns_dict, feat_inputs_dict,is_linear=True, init_std=0.0001,seed=1024,l2_reg=0.0001):
    cat_feature_columns = list(
        filter(lambda x: isinstance(x, Categorical_Feature), feat_columns_dict)) if feat_columns_dict else []
    if is_linear:
        cat_embedding = {feat.embedding_name: Embedding(feat.vocabulary_size, 1,
                                                           embeddings_initializer=RandomNormal(
                                                               mean=0.0, stddev=init_std, seed=seed),
                                                           embeddings_regularizer=l2(l2_reg),
                                                           name='linear_emb_' + feat.embedding_name)(feat_inputs_dict[feat.name])
                            for feat in cat_feature_columns}
    else:
        cat_embedding = {feat.embedding_name: Embedding(feat.vocabulary_size, feat.embedding_dim,
                                                           embeddings_initializer=RandomNormal(
                                                               mean=0.0, stddev=init_std, seed=seed),
                                                           embeddings_regularizer=l2(l2_reg),
                                                           name='dnn_emb_' + feat.embedding_name)(feat_inputs_dict[feat.name])
                            for feat in cat_feature_columns}
    return cat_embedding

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

def get_numeric_input(features, feature_columns):
    numeric_feature_columns = list(filter(lambda x: isinstance(x, Numeric_Feat), feature_columns)) if feature_columns else []
    numeric_input_list = []
    for fc in numeric_feature_columns:
        numeric_input_list.append(features[fc.name])
    return numeric_input_list

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
               name=None):
    return tf.reduce_max(input_tensor,
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
