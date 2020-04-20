# -*- coding:utf-8 -*-

"""
Author:
    Qingzhongwen, VX: ziqingxiansheng
"""

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from deep_models.keras_models.layers.core import PredictionLayer, DNN
from deep_models.keras_models.layers.linear import Linear
from deep_models.keras_models.layers.cin_layer import CIN
from deep_models.keras_models.layers.utils import combined_dnn_input,get_feature_inputs_dict, get_cat_embeddings, dict_to_list, get_linear_inputs
from deep_models.keras_models.layers.utils import concat_func, add_func

# Keras functional API
def Keras_xDeepFM(all_feature_columns_dict_list, dnn_hidden_units=(1024, 512, 256), cin_layer_size=(128, 128,),
                 l2_reg_linear=0.0001, l2_reg_embedding=0.0001, l2_reg_dnn=0.0001,cin_split_half=True,
                 init_std=0.0001, seed=1024, dnn_dropout=0.4, activation='selu',l2_reg_cin=0.00001,
                 dnn_use_bn=True, use_bias=True, task='binary'):

    # 构建输入tf.keras.Input函数，Input()用于实例化Keras张量 返回features：tensor.
    # Tensor shape：(None, 1)
    numeric_inputs_dict,categorical_inputs_dict = get_feature_inputs_dict(all_feature_columns_dict_list)

    # 将 Embedding层 与 Input层 进行拼接 shape：(None, 1, embedding_dim)
    linear_cat_embeddings_dict = get_cat_embeddings(all_feature_columns_dict_list, categorical_inputs_dict, is_linear=True,
                                               init_std=init_std,seed=seed,l2_reg=l2_reg_embedding)
    dnn_cat_embeddings_dict = get_cat_embeddings(all_feature_columns_dict_list, categorical_inputs_dict, is_linear=False,
                                            init_std=init_std,seed=seed,l2_reg=l2_reg_embedding)

    numeric_inputs_list = dict_to_list(numeric_inputs_dict)
    categorical_inputs_list = dict_to_list(categorical_inputs_dict)
    inputs_list = categorical_inputs_list + numeric_inputs_list
    # shape：(None, 1, 8)
    dnn_cat_embeddings_list = dict_to_list(dnn_cat_embeddings_dict)

    # linear_logit Linear
    linear_inputs = get_linear_inputs(numeric_inputs_dict,linear_cat_embeddings_dict)
    linear_logit = Linear(l2_reg_linear, mode=2, use_bias=use_bias)(linear_inputs)

    # shape：(None, 7, 8)
    fm_inputs = concat_func(dnn_cat_embeddings_list, axis=1)

    # shape：(None, 58)
    dnn_input = combined_dnn_input(dnn_cat_embeddings_list, numeric_inputs_list)
    # shape：(None, 256)
    dnn_output = DNN(dnn_hidden_units, activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed)(dnn_input)

    # shape：(None, 1)
    dnn_logit = Dense(1, use_bias=False, activation=None)(dnn_output)

    # shape：(None, 1)
    final_logit = add_func([linear_logit, dnn_logit])

    if len(cin_layer_size) > 0:
        xFM_out = CIN(cin_layer_size, activation,
                       cin_split_half, l2_reg_cin, seed)(fm_inputs)
        xFM_logit = tf.keras.layers.Dense(1, activation=None, )(xFM_out)
        final_logit = add_func([final_logit, xFM_logit])

    output = PredictionLayer(task)(final_logit)
    model = Model(inputs=inputs_list, outputs=output)
    return model
