from .utils import reduce_sum

from .core import PoolingLayer, Similarity, LabelAwareAttention, CapsuleLayer,SampledSoftmaxLayer,EmbeddingIndex, \
    DNN, LocalActivationUnit, PredictionLayer
from .interaction import DotAttention, ConcatAttention, SoftmaxWeightedSum, AttentionSequencePoolingLayer, SelfAttention,\
    SelfMultiHeadAttention, UserAttention
from .sequence import DynamicMultiRNN
import tensorflow as tf
from .activation import Dice
from .interaction import (CIN, FM, AFMLayer, BiInteractionPooling, CrossNet,
                          InnerProductLayer, InteractingLayer,
                          OutterProductLayer, FGCNNLayer,SENETLayer,BilinearInteraction,
                          FieldWiseBiInteraction)
from .normalization import LayerNormalization
from .sequence import (AttentionSequencePoolingLayer, BiasEncoding, BiLSTM,
                       KMaxPooling, SequencePoolingLayer,WeightedSequenceLayer,
                       Transformer, DynamicGRU)
from .utils import NoMask, Hash, Linear, Add, sampledsoftmaxloss

custom_objects = {'tf': tf,
                  'InnerProductLayer': InnerProductLayer,
                  'OutterProductLayer': OutterProductLayer,
                  'DNN': DNN,
                  'PredictionLayer': PredictionLayer,
                  'FM': FM,
                  'AFMLayer': AFMLayer,
                  'CrossNet': CrossNet,
                  'BiInteractionPooling': BiInteractionPooling,
                  'LocalActivationUnit': LocalActivationUnit,
                  'Dice': Dice,
                  'SequencePoolingLayer': SequencePoolingLayer,
                  'AttentionSequencePoolingLayer': AttentionSequencePoolingLayer,
                  'CIN': CIN,
                  'InteractingLayer': InteractingLayer,
                  'LayerNormalization': LayerNormalization,
                  'BiLSTM': BiLSTM,
                  'Transformer': Transformer,
                  'NoMask': NoMask,
                  'BiasEncoding': BiasEncoding,
                  'KMaxPooling': KMaxPooling,
                  'FGCNNLayer': FGCNNLayer,
                  'Hash': Hash,
                  'Linear':Linear,
                  'DynamicGRU': DynamicGRU,
                  'SENETLayer':SENETLayer,
                  'BilinearInteraction':BilinearInteraction,
                  'WeightedSequenceLayer':WeightedSequenceLayer,
                  'Add':Add,
                  'FieldWiseBiInteraction':FieldWiseBiInteraction
                  }


_custom_objects = {'PoolingLayer': PoolingLayer,
                   'Similarity': Similarity,
                   'LabelAwareAttention': LabelAwareAttention,
                   'CapsuleLayer': CapsuleLayer,
                   'reduce_sum':reduce_sum,
                   'SampledSoftmaxLayer':SampledSoftmaxLayer,
                   'sampledsoftmaxloss':sampledsoftmaxloss,
                   'EmbeddingIndex':EmbeddingIndex,
                   'DotAttention':DotAttention,
                   'ConcatAttention':ConcatAttention,
                   'SoftmaxWeightedSum':SoftmaxWeightedSum,
                   'AttentionSequencePoolingLayer':AttentionSequencePoolingLayer,
                   'SelfAttention':SelfAttention,
                   'SelfMultiHeadAttention':SelfMultiHeadAttention,
                   'UserAttention':UserAttention,
                    'DynamicMultiRNN':DynamicMultiRNN
                   }

custom_objects = dict(custom_objects, **_custom_objects)
