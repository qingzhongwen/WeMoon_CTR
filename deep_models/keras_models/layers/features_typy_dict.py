# -*- coding:utf-8 -*-

from collections import namedtuple

class Categorical_Feature(namedtuple('CatFeat',
                            ['name', 'vocabulary_size', 'dimension','embedding_dim', 'dtype', 'embedding_name'])):
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, dimension=1, embedding_dim=4, dtype="int32", embedding_name=None):
        if embedding_name is None:
            embedding_name = name
        return super(Categorical_Feature, cls).__new__(cls, name, vocabulary_size, dimension, embedding_dim, dtype,
                                                       embedding_name)

class Numeric_Feat(namedtuple('NumFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(Numeric_Feat, cls).__new__(cls, name, dimension, dtype)


