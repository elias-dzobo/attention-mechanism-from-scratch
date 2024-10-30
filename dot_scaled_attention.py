# In Tensorflow and Keras
import tensorflow
from tensorflow import matmul, math, cast, float32
from tensorflow.python.keras.layers import Layer 
from tensorflow.python.keras.backend import softmax 
from numpy import random

class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)

    def call(self, queries, keys, values, d_k, mask=None):
        scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))

        if mask is not None:
            scores += -1e9 * mask 

        weights = softmax(scores)

        return matmul(weights, values)
    
## test 
d_k = 64
d_v = 64
batch_size = 64 
input_seq_len = 5 

queries = random.random((batch_size, input_seq_len, d_k))
keys = random.random((batch_size, input_seq_len, d_k))
values = random.random((batch_size, input_seq_len, d_v))

attention = DotProductAttention()
print(attention(queries, keys, values, d_k)) 


