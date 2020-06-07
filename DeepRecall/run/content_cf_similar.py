# -*- coding:utf-8 -*-

"""
Author:
    Qingzhongwen, VX: ziqingxiansheng
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR))
#可以选择用sys.path.insert(0,‘/path’)，这样新添加的目录会优先于其他目录被import检查
from offline import SparkSessionBase
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.feature import VectorAssembler
import pandas as pd

cols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
       '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
       '23', '24', '25', '26', '27', '28', '29', '30', '31']

class ItemSimilarModel(SparkSessionBase):
    SPARK_APP_NAME = "Similar"
    SPARK_URL = "yarn"

    ENABLE_HIVE_SUPPORT = True

    def __init__(self):
        self.spark = self._create_spark_session()


# create 'item_similar', 'similar'
# desc 'item_similar'
# scan 'item_similar'
# get 'item_similar' , '3416'
# 保存至hbase
def save_hbase(partitions):
    import happybase
    pool = happybase.ConnectionPool(size=3, host='hadoop-master')

    with pool.connection() as conn:
        article_similar = conn.table('item_similar')
        for row in partitions:
            if row.datasetA.movie_id == row.datasetB.movie_id:
                pass
            else:
                article_similar.put(str(row.datasetA.movie_id).encode(),
                                    {'similar:{}'.format(row.datasetB.movie_id).encode(): b'%0.4f' % (
                                        row.EuclideanDistance)})

if __name__ == '__main__':

    ism = ItemSimilarModel()
    item_embeddings = pd.read_csv('./item_embeddings.csv')
    item_df = ism.spark.createDataFrame(item_embeddings)

    #将YoutubeDNN模型导出的32维movieid向量转化成LSH所需的vector格式
    embedding_vecAssembler = VectorAssembler(inputCols=cols, outputCol="embeddings").transform(item_df)
    embedding_vecAssembler.registerTempTable('temptable')
    embedding_Vectors = ism.spark.sql("select movie_id, embeddings from temptable")

    # 计算相似的item
    brp = BucketedRandomProjectionLSH(inputCol='embeddings', outputCol='similar', numHashTables=4.0, bucketLength=10.0)
    model = brp.fit(embedding_Vectors)
    similar = model.approxSimilarityJoin(embedding_Vectors, embedding_Vectors, 2.0, distCol='EuclideanDistance')

    #数据入库
    similar.foreachPartition(save_hbase)


