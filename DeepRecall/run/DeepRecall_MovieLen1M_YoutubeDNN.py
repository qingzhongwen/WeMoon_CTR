import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from ..models import *
from ..layers import sampledsoftmaxloss
from ..tools.utils import SparseFeat, VarLenSparseFeat
from ..tools.preprocess import gen_data_set, gen_model_input


if __name__ == '__main__':
    print("start!!!")
    data_path = "./"

    unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
    user = pd.read_csv(data_path + 'ml-1m/users.dat', sep='::', header=None, names=unames)
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(data_path + 'ml-1m/ratings.dat', sep='::', header=None, names=rnames)
    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(data_path + 'ml-1m/movies.dat', sep='::', header=None, names=mnames)

    data = pd.merge(pd.merge(ratings, movies), user)  # .iloc[:10000]

    # data = pd.read_csvdata = pd.read_csv("./movielens_sample.txt")
    sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zip", ]
    SEQ_LEN = 50
    negsample = 0

    # 1.Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`
    # 1. 首先对于数据中的特征进行ID化编码，然后使用 `gen_date_set` and `gen_model_input`来生成带有用户历史行为序列的特征数据
    features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']
    feature_max_idx = {}
    for feature in features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1

    user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')
    item_profile = data[["movie_id"]].drop_duplicates('movie_id')
    user_profile.set_index("user_id", inplace=True)
    user_item_list = data.groupby("user_id")['movie_id'].apply(list)

    train_set, test_set = gen_data_set(data, negsample)
    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)

    # 2.count #unique features for each sparse field and generate feature config for sequence feature
    # 2. 配置一下模型定义需要的特征列，主要是特征名和embedding词表的大小
    embedding_dim = 32

    user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], 16),
                            SparseFeat("gender", feature_max_idx['gender'], 16),
                            SparseFeat("age", feature_max_idx['age'], 16),
                            SparseFeat("occupation", feature_max_idx['occupation'], 16),
                            SparseFeat("zip", feature_max_idx['zip'], 16),
                            VarLenSparseFeat(SparseFeat('hist_movie_id', feature_max_idx['movie_id'], embedding_dim,
                                                        embedding_name="movie_id"), SEQ_LEN, 'mean', 'hist_len'),
                            ]

    item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], embedding_dim)]

    # 3.Define Model and train
    # 3. 定义一个YoutubeDNN模型，分别传入用户侧特征列表`user_feature_columns`和物品侧特征列表`item_feature_columns`。然后配置优化器和损失函数，开始进行训练。
    #keras.backend.set_learning_phase(0) # train mode
    # keras.backend.set_learning_phase(1) # predict mode
    # 之所以这么区分，是因为某些层在预测和训练时不同
    K.set_learning_phase(True)

    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()

    model = YoutubeDNN(user_feature_columns, item_feature_columns, num_sampled=100,
                       user_dnn_hidden_units=(128, 64, embedding_dim))
    model.compile(optimizer="adam", loss=sampledsoftmaxloss)  # "binary_crossentropy")
    history = model.fit(train_model_input, train_label,  # train_label,
                        batch_size=512, epochs=2, verbose=1, validation_split=0.0, )

    # 4. Generate user features for testing and full item features for retrieval
    # 4. 训练完整后，由于在实际使用时，我们需要根据当前的用户特征实时产生用户侧向量，并对物品侧向量构建索引进行近似最近邻查找。
    # 这里由于是离线模拟，所以我们导出所有待测试用户的表示向量，和所有物品的表示向量。
    test_user_model_input = test_model_input
    #获取每个movie_id
    all_item_model_input = {"movie_id": item_profile['movie_id'].values, }

    # 以下两行是deepmatch中的通用使用方法，分别获得用户向量模型和物品向量模型
    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)
    # print(user_embedding_model)
    # print(item_embedding_model)

    print('='*80)
    # 输入对应的数据拿到对应的向量
    user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
    # user_embs = user_embs[:, i, :]  # i in [0,k_max) if MIND
    item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

    print(user_embs.shape)
    print(item_embs.shape)

    user_emb = pd.DataFrame(user_embs)
    print(user_emb.head())
    item_emb = pd.DataFrame(item_embs)
    item_emb['index'] = item_emb.index
    print(item_emb.head())

    item_profile.reset_index(drop=True,inplace=True)
    item_profile['index'] = item_profile.index

    item_embeddings = pd.merge(item_profile, item_emb, how='left', on='index')
    item_embeddings.drop(['index'], axis=1, inplace=True)
    print(item_embeddings.head())
    # item_embeddings.to_csv("./item_embeddings.csv", index=False)


