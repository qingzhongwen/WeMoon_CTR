# -*- coding:utf-8 -*-

from DeepRecall.tools.utils import input_from_feature_columns, build_input_features, combined_dnn_input, create_embedding_matrix, get_item_embedding
from DeepRecall.layers.utils import NoMask
from tensorflow.python.keras.models import Model
from DeepRecall.layers.core import DNN, PoolingLayer, SampledSoftmaxLayer, EmbeddingIndex


def YoutubeDNN(user_feature_columns, item_feature_columns, num_sampled=5,
               user_dnn_hidden_units=(64, 32),
               dnn_activation='relu', dnn_use_bn=False,
               l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001, seed=1024, ):
    """Instantiates the YoutubeDNN Model architecture.

    :param user_feature_columns: An iterable containing user's features used by  the model.
    :param item_feature_columns: An iterable containing item's features used by  the model.
    :param num_sampled: int, the number of classes to randomly sample per batch.
    :param user_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of user tower
    :param dnn_activation: Activation function to use in deep net
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :return: A Keras model instance.

    """

    if len(item_feature_columns) > 1:
        raise ValueError("Now YoutubeNN only support 1 item feature like item_id")
    item_feature_name = item_feature_columns[0].name
    item_vocabulary_size = item_feature_columns[0].vocabulary_size

    #构建User和 item 的 Embedding dict
    embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns, l2_reg_embedding,
                                                    init_std, seed, prefix="")

    #构建 User Input dict
    user_features = build_input_features(user_feature_columns)
    user_inputs_list = list(user_features.values())
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features,
                                                                                   user_feature_columns,
                                                                                   l2_reg_embedding, init_std, seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict
                                                                                   )
    user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)


    # 构建 item Input dict
    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())

    user_dnn_out = DNN(user_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, seed, )(user_dnn_input)

    print(item_features[item_feature_name])
    print(list(range(item_vocabulary_size)))
    item_index = EmbeddingIndex(list(range(item_vocabulary_size)))(item_features[item_feature_name])

    item_embedding_matrix = embedding_matrix_dict[
        item_feature_name]
    print(item_embedding_matrix(item_index))
    item_embedding_weight = NoMask()(item_embedding_matrix(item_index))

    pooling_item_embedding_weight = PoolingLayer()([item_embedding_weight])

    # tf.nn.sampled_softmax_loss,在类别很多的情况下训练softmax分类器的高效的方法。
    # 注意：仅在训练时使用这一采样操作，测试时还是用全部的类别
    output = SampledSoftmaxLayer(num_sampled=num_sampled)(
        [pooling_item_embedding_weight, user_dnn_out, item_features[item_feature_name]])
    model = Model(inputs=user_inputs_list + item_inputs_list, outputs=output)

    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("user_embedding", user_dnn_out)

    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("item_embedding",
                      get_item_embedding(pooling_item_embedding_weight, item_features[item_feature_name]))

    return model
