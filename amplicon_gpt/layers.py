import os
from collections import namedtuple
import tensorflow as tf
from amplicon_gpt.initializers import UnitUniform
import tensorflow_models as tfm

def create_tf_func(func, *args, jit_compile=True, **kwargs):
    return tf.function(
        lambda x: func(x, *args),
        jit_compile=jit_compile,
    )

def create_concrete_tf_func(func, *args, tensor_spec=None, **kwargs):
    return create_tf_func(func, *args).get_concrete_function(tensor_spec)

def create_tf_with_training_flag(func, tensor_spec, **kwargs):
    return {
        True: create_concrete_tf_func(func, tensor_spec=tensor_spec, training=True, **kwargs),
        False: create_concrete_tf_func(func, tensor_spec=tensor_spec, training=False, **kwargs)
    }

def get_tf_training_flag_version(dict, training):
    return dict[training]

def run_tf_with_training_flag_dict(dict, input, training):
    return get_tf_training_flag_version(dict, training)(input)


@tf.keras.saving.register_keras_serializable(package="amplicon_gpt", name="NucleotideSequenceEmbedding")
class NucleotideSequenceEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, dropout, **kwargs):
        super().__init__(name="nucleotide_sequence_embedding", **kwargs)
        self.embedding_dim = embedding_dim
        self.lstm = tf.keras.layers.TimeDistributed(tf.keras.layers.LSTM(embedding_dim, dropout=dropout, return_sequences=True))
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.norm = tf.keras.layers.LayerNormalization()
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1, activation='relu'),
            tf.keras.layers.Flatten(),
        ])

    def build(self, input_shape):
        dense_shape = input_shape[0], input_shape[2], input_shape[3]
        vec_layer_spec = tf.TensorSpec((dense_shape), dtype=tf.float32)
        def _vectorize_dense(input):
            output = tf.transpose(input, perm=[0,2,1])
            return self.dense(output)
        self.vec_layer_dict = create_tf_with_training_flag(_vectorize_dense, vec_layer_spec)

    def call(self, input, training=False):
        output = self.lstm(input)
        output = self.norm(output)
        output = self.dropout(output, training)

        output = tf.transpose(output, perm=[1,0,2,3])
        tf_vec_func = get_tf_training_flag_version(self.vec_layer_dict, training)
        output = tf.vectorized_map(tf_vec_func, output, fallback_to_while_loop=False)
        output = tf.transpose(output, perm=[1,0,2])
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
                "embedding_dim": self.embedding_dim,
                "dropout": self.dropout
        })
        return config
        
@tf.keras.saving.register_keras_serializable(package="amplicon_gpt", name="PositionEncoder")
class SampleEncoder(tf.keras.layers.Layer):
    def __init__(self, nucleotide_embedding_dim, dropout, num_enc_layers, num_heads, dff, norm_first, **kwargs):
        super().__init__(name="sample_encoder", **kwargs)
        self.nucleotide_embedding_dim = nucleotide_embedding_dim
        self.asv_pos_emb = tf.keras.layers.LSTM(nucleotide_embedding_dim, dropout=dropout, return_sequences=True)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

        self.norm = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

        self.encoding_blocks = tfm.nlp.models.TransformerEncoder(
            num_layers=num_enc_layers,
            num_attention_heads=8,
            intermediate_size=2048,
            dropout_rate=dropout,
            norm_first=True,
            activation='gelu',
        )
            
    
    def build(self, input_shape):
        def pad(x):
            padddings = tf.constant([[128, 0], [0,0]])
            output = tf.pad(x, padddings)
            output = tf.strided_slice(output, begin=[-129,0], end=[-1, input_shape[2]], strides=[1,1])
            return output
        pad_spec = tf.TensorSpec(shape=input_shape[1:], dtype=tf.float32)
        self.pad = create_concrete_tf_func(pad, tensor_spec=pad_spec)
        
        causal_mask = tf.cast(1 - tf.linalg.band_part(tf.ones((128, 128)), -1, 0), dtype=tf.bool)
        encoding_spec = tf.TensorSpec(shape=[input_shape[0], 128, input_shape[2]], dtype=tf.float32)
        self.encoding_block_dict = create_tf_with_training_flag(lambda x: self.encoding_blocks(x, attention_mask=causal_mask), encoding_spec)

    def call(self, input, training=False):
        print(training)
        output = self.asv_pos_emb(input)
        output = self.norm(output)
        output = self.dropout(output, training=training)

        output = tf.vectorized_map(self.pad, output, fallback_to_while_loop=False)
        output = self.norm2(output)
        output = self.dropout2(output, training=training)
        
        return run_tf_with_training_flag_dict(self.encoding_block_dict, output, training)
            
    def get_config(self):
        config = super().get_config()
        config.update({
                "dropout": self.dropout
        })
        return config
    
@tf.keras.saving.register_keras_serializable(package="amplicon_gpt", name="UniFracEncoder")
class UniFracEncoder(tf.keras.layers.Layer):
    def __init__(self, dff, **kwargs):
        super().__init__(name="unifrac_embedding", **kwargs)
        self.ff = tf.keras.layers.Dense(128, activation='relu')
        self.norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense = tf.keras.layers.Dense(1)
    
    def build(self, input_shape):

        def vec_layer(input, training=None):
            output = self.ff(input)
            output = self.norm(output)
            output = self.dropout(output, training=training)
            return output
        tensor_spec = tf.TensorSpec(shape=input_shape[1:], dtype=tf.float32)
        self.vectorize_tf_dict = create_tf_with_training_flag(vec_layer, tensor_spec=tensor_spec)

    def call(self, input, training=False):
        tf_vec_func = get_tf_training_flag_version(self.vectorize_tf_dict, training)
        output = tf.vectorized_map(tf_vec_func, input, fallback_to_while_loop=False)
        return tf.squeeze(self.dense(output))

            