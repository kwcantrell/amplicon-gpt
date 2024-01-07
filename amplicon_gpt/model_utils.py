import os
import numpy as np
import pandas as pd
from biom import load_table
import tensorflow as tf
import keras_nlp
from tensorflow_models import nlp # need for PositionEmbedding without cannot load base_model
from amplicon_gpt.losses import unifrac_loss_var, _pairwise_distances # need for unifrac_loss_var without cannot load base_model
from amplicon_gpt.losses import regression_loss_variance, regression_loss_difference_in_means, regression_loss_combined, regression_loss_normal
from amplicon_gpt.layers import NucleotideSequenceEmbedding, SampleEncoder, UniFracEncoder

# tf.config.optimizer.set_jit(True)
# physical_devices = tf.config.list_physical_devices('GPU')
# for device in physical_devices:
#   tf.config.experimental.set_memory_growth(device, True)
MAX_SEQ = 1600

def create_conv_config(num_filters=32, kernel_size=5, stride=1, padding='valid'):
    return (num_filters, kernel_size, stride, padding)

def append_config_num(config, num):
    config['name'] = f"{config['name']}_num"
    return config

"""

"""

def transfer_learn_base(sequence_tokenizer, lstm_seq_out, batch_size, max_num_per_seq, dropout, root_path, load_prev_path=False, **kwargs):   
    d_model = 64
    dff = 512
    num_heads = 6
    num_enc_layers = 4
    norm_first = False
    
    @tf.keras.saving.register_keras_serializable()
    class UniFrac(tf.keras.Model):
        def __init__(
            self,
            d_model,
            dff,
            num_heads,
            num_enc_layers,
            norm_first=False,
            name="unifrac",
            **kwargs
        ):
            """
            need sequence tokenizer
                 batch_size
                 dropout
            """
            super().__init__(name=name, **kwargs)
            self.d_model = d_model
            self.dff = dff
            self.num_heads = num_heads
            self.num_enc_layers = num_enc_layers
            
            self.seq_token = sequence_tokenizer
            self.embedding = tf.keras.layers.Embedding(5, d_model, input_length=100,
                                                       input_shape=[batch_size, None, 100])
            self.nuc = NucleotideSequenceEmbedding(d_model, dropout)
            self.samp = SampleEncoder(d_model, dropout, num_enc_layers, num_heads, dff, norm_first)
            self.lstm = tf.keras.layers.LSTM(d_model, dropout=dropout, return_sequences=True)
            self.unif = UniFracEncoder(dff)

        def call(self, inputs, training=None):
            @tf.function(
                input_signature=[tf.TensorSpec(
                    tf.TensorShape([batch_size, None, 1]), dtype=tf.string)],
                reduce_retracing=True,
                jit_compile=False
            ) 
            def _call(input):   
                # output = input.to_tensor()
                output = self.seq_token(input)
                output = self.embedding(output)
                output = self.nuc(output)
                output = self.samp(output)
                output = self.lstm(output)
                return self.unif(output)
            return _call(inputs)
        
    if load_prev_path:
        model = tf.keras.models.load_model(os.path.join(root_path, 'encoder.keras'))
        return model
        
    return UniFrac(d_model, dff, num_heads, num_enc_layers)
    
    
#     input = tf.keras.Input(shape=(None,1), batch_size=batch_size,  name='model_input',
#                            dtype=tf.string)
    
#     output = sequence_tokenizer(input)
#     encoding_blocks = [
#        NucleotideSequenceEmbedding(d_model, dropout), 
#        SampleEncoder(d_model, dropout, num_enc_layers, num_heads, dff, norm_first),
#        tf.keras.layers.LSTM(128, dropout=dropout, return_sequences=True),
#        UniFracEncoder()
#     ]
    
#     output = tf.keras.Sequential(encoding_blocks)(output)
#     model = tf.keras.Model(inputs=input, outputs=output)    
    
    # return model

@tf.keras.saving.register_keras_serializable(package="Scale16s", name="MAE")
class MAE(tf.keras.metrics.Metric):
    def __init__(self, name='mae_loss', dtype=tf.float32):
        super().__init__(name=name, dtype=dtype)
        self.loss = self.add_weight(name='rl', initializer='zero', dtype=tf.float32)
        self.i = self.add_weight(name='i', initializer='zero', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.loss.assign_add(tf.reduce_sum(tf.abs(_pairwise_distances(y_pred)-y_true)))
        self.i.assign_add(tf.constant(8.0, dtype=tf.float32))

    def result(self):
        return self.loss / self.i
        
def compile_model(model):
    lr = 0.0001#tf.keras.optimizers.schedules.ExponentialDecay(0.0001, decay_steps=150000, decay_rate=0.9, staircase=True)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, epsilon=1e-7)
    model.compile(optimizer=optimizer,loss=unifrac_loss_var, metrics=[MAE()])
    return model


def load_full_base_model(base_model_path, **kwargs):
    base_model = tf.keras.models.load_model(base_model_path)
    base_model.trainable = False
    return base_model

def get_base_model(base_model_path, **kwargs):
    """
    Returns the base model input layer and the output of the 'community level' encoder (i.e. the transformer encoder block).
    This will also disable all trainable parameters.
    """
    base_model = tf.keras.models.load_model(base_model_path)
    base_model.trainable = False
    input = base_model.inputs[0] # base model only has one input however, .inputs returns list
    base_output = base_model.get_layer('base_encoder_block_3').output
    return input, base_output

def _add_feature_regression_module(input, microbe_mask, lstm_seq_out, dropout, conv_config, num_enc_layers=4, output_units=1):
    num_heads = 4
    dff = 64
    norm_first = False
    encoders = [TransformerEncoder(num_heads=num_heads, dropout=dropout,
                                       activation='gelu', intermediate_dim=dff, normalize_first=norm_first,
                                       name=f'feature_encoder_block_{i}')
                    for i in range(num_enc_layers)]
    output = encoders[0](input, microbe_mask)
    for encoder in encoders[1:]:
        output = encoder(output, microbe_mask, training=True)
    output = tf.keras.layers.LSTM(lstm_seq_out, dropout=dropout, name='feature_lstm')(output, mask=microbe_mask, training=True)
    output = tf.keras.layers.Dropout(dropout)(output, training=True)
    output = tf.expand_dims(output, axis=-1)
    for i, (conv_filter, kernel_size, stride, padding) in enumerate(conv_config):
        output = tf.keras.layers.Conv1D(conv_filter, kernel_size, 
                                        strides=stride, padding=padding, activation='relu',
                                        name=f'feature_conv_{i}')(output)
        output = tf.keras.layers.Dropout(dropout)(output, training=True)
        output = tf.keras.layers.MaxPool1D(2)(output)
    output = tf.keras.layers.Flatten(name='feature_flatten')(output)
    return tf.keras.layers.Dense(output_units, use_bias=False, name='feature_regression_output')(output)

def transfer_learn_feature_regression(load_prev_path, lstm_seq_out, dropout, root_path, num_enc_layers, use_ema=False, ema_momentum=None, **config):
    if load_prev_path:
        model = tf.keras.models.load_model(os.path.join(root_path, 'model.keras'))
        lr = tf.keras.optimizers.schedules.ExponentialDecay(0.0005, decay_steps=10000, decay_rate=0.99, staircase=True)
        if use_ema:
            optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, epsilon=1e-8, ema_momentum=ema_momentum, use_ema=use_ema, ema_overwrite_frequency=None)
        else:
            optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, epsilon=1e-8)
        model.compile(optimizer=optimizer,loss=loss, metrics=[MAE()])
        return model

    conv_config = [
        create_conv_config(num_filters=32, kernel_size=5, stride=1, padding='same'),
        # create_conv_config(num_filters=64, kernel_size=2, stride=2, padding='same'),
        create_conv_config(num_filters=16, kernel_size=5, stride=1, padding='same'),
        # create_conv_config(num_filters=128, kernel_size=2, stride=2, padding='valid')
    ]
    loss = regression_loss_variance

    @tf.keras.saving.register_keras_serializable(package="Scale16s", name="MAE")
    class MAE(tf.keras.metrics.Metric):
        def __init__(self, name='mae_loss', dtype=tf.float32):
            super().__init__(name=name)
            self.loss = self.add_weight(name='rl', initializer='zero', dtype=tf.float32)
            self.i = self.add_weight(name='i', initializer='zero', dtype=tf.float32)

        def update_state(self, y_true, y_pred,  **kwargs):
            self.loss.assign_add(tf.reduce_sum(tf.abs(y_pred-y_true)))
            self.i.assign_add(16.0)

        def result(self):
            return self.loss / self.i
        
    input, base_output = get_base_model(**config)
    microbe_mask = tf.cast(tf.not_equal(input[:, :, 0], 0), tf.bool)
    output = _add_feature_regression_module(base_output, microbe_mask,
                                       lstm_seq_out, dropout, conv_config, 
                                       num_enc_layers, output_units=1)
    model = tf.keras.Model(inputs=input, outputs=output)

    lr = tf.keras.optimizers.schedules.ExponentialDecay(0.0001, decay_steps=10000, decay_rate=0.99, staircase=True)
    if use_ema:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, epsilon=1e-8, ema_momentum=ema_momentum, use_ema=use_ema, ema_overwrite_frequency=None)
    else:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, epsilon=1e-8)
    model.compile(optimizer=optimizer,loss=loss, metrics=[MAE()])
    return model


def _add_feature_classification_module(input, microbe_mask, lstm_seq_out, dropout, conv_config, num_enc_layers=4, output_units=1):
    num_heads = 6
    dff = 128
    norm_first = False
    encoders = [TransformerEncoder(num_heads=num_heads, dropout=dropout,
                                       activation='gelu', intermediate_dim=dff, normalize_first=norm_first,
                                       name=f'classification_encoder_block_{i}')
                    for i in range(num_enc_layers)]
    output = encoders[0](input, microbe_mask)
    for encoder in encoders[1:]:
        output = encoder(output, microbe_mask)
    output = tf.keras.layers.LSTM(lstm_seq_out, dropout=dropout, name='classification_lstm')(output, mask=microbe_mask, training=True)
    output = tf.expand_dims(output, axis=-1)
    for (conv_filter, kernel_size, stride, padding) in conv_config:
        output = tf.keras.layers.Conv1D(conv_filter, kernel_size, strides=stride, padding=padding)(output)
    output = tf.keras.layers.Flatten(name='classification_flatten')(output)   
    return tf.keras.layers.Dense(output_units, activation='sigmoid', name='classification_output')(output)

def transfer_learn_feature_classification(continue_training, lstm_seq_out, dropout, root_path, num_enc_layers, use_ema=False, ema_momentum=None, **config):
    METRICS = [
        tf.keras.metrics.BinaryCrossentropy(name='cross entropy'),  # same as model's loss
        tf.keras.metrics.MeanSquaredError(name='Brier score'),
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]

    if continue_training:
        model = tf.keras.models.load_model(os.path.join(root_path, 'model.keras'))
        return model

    conv_config = [create_conv_config(num_filters=256, kernel_size=3, stride=1, padding='same'),
                    create_conv_config(num_filters=64, kernel_size=2, stride=2, padding='valid'),
                ]
    
    input, base_output = get_base_model(**config)
    microbe_mask = tf.cast(tf.not_equal(input[:, :, 0], 0), tf.bool)
    output = _add_feature_classification_module(base_output, microbe_mask,
                                       lstm_seq_out, dropout, conv_config, 
                                       num_enc_layers, output_units=1)
    model = tf.keras.Model(inputs=input, outputs=output)
    lr = tf.keras.optimizers.schedules.ExponentialDecay(0.0005, decay_steps=70000, decay_rate=0.99, staircase=True)
    if use_ema:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, epsilon=1e-8, ema_momentum=ema_momentum, use_ema=use_ema, ema_overwrite_frequency=None)
    else:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, epsilon=1e-8)
    loss = tf.keras.losses.BinaryCrossentropy(reduction='sum_over_batch_size')
    model.compile(optimizer=optimizer,loss=loss, metrics=METRICS)
    return model