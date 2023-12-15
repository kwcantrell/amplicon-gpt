import os
import tensorflow as tf
import keras_nlp
from amplicon_gpt.initializers import UnitUniform

nucleotide_embedding_dim=256
nuc_norm_epsilon=1e-5
d_model = 128
dff = 512
num_heads = 6
num_enc_layers = 4
lstm_nuc_out = 128
lstm_seq_out = 128
emb_vec = 32
norm_first = False
conv_1_filter = 256
conv_2_filter = 64

@tf.keras.saving.register_keras_serializable(package="amplicon_gpt", name="NucleotideSequenceEmbedding")
class NucleotideSequenceEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, dropout, **kwargs):
        super().__init__(name="nucleotide_sequence_embedding", **kwargs)
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        
        self.embedding = tf.keras.layers.Embedding(5, embedding_dim, input_length=150, mask_zero=False)
        self.pos_embedding = keras_nlp.layers.PositionEmbedding(sequence_length=150)
        self.lstm = tf.keras.layers.LSTM(32, return_sequences=False, dropout=dropout)

    def call(self, input, training=False):
        emb = self.embedding(input)
        output = emb + self.pos_embedding(emb)

        # def map_lstm(sequences, mask):
        def map_lstm(sequences):
            seq_in = tf.TensorArray(tf.float32, size=16, dynamic_size=False, colocate_with_first_write_call=False)
            seq_in = seq_in.unstack(sequences)
            seq_out = tf.TensorArray(tf.float32, size=16, dynamic_size=False, colocate_with_first_write_call=False)
            # mask_array = tf.TensorArray(tf.bool, size=16, dynamic_size=False)
            # mask_array = mask_array.unstack(mask)
        
            for i in tf.range(16):
                seq_out = seq_out.write(
                    i,
                    self.lstm(seq_in.read(i))
                )
            return seq_out.stack() 
        # sequence_masks = tf.not_equal(input, 0)
        return map_lstm(output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
                "embedding_dim": self.embedding_dim,
                "dropout": self.dropout
        })
        return config
    
@tf.keras.saving.register_keras_serializable(package="amplicon_gpt", name="ASVEncoder")
class ASVEncoder(tf.keras.layers.Layer):
    def __init__(self, dropout, **kwargs):
        super().__init__(name="asv_sequence_embedding", **kwargs)
        self.dropout = dropout
        self.asv_pos_emb = keras_nlp.layers.PositionEmbedding(sequence_length=1600)
        self.asv_norm = tf.keras.layers.LayerNormalization(epsilon=nuc_norm_epsilon)
        self.add = tf.keras.layers.Add()
        self.asv_dropout = tf.keras.layers.Dropout(dropout)
        self.encoder_blocks = [
            keras_nlp.layers.TransformerEncoder(num_heads=num_heads, dropout=dropout,
                    activation='gelu', intermediate_dim=dff, normalize_first=norm_first,
                    name=f'base_encoder_block_{i}')
        for i in range(num_enc_layers)]

    def call(self, input, training=False):
        mask = tf.reduce_any(tf.not_equal(input, 0), axis=2)

        asv_pos = self.asv_pos_emb(input)
        asv_pos = self.asv_norm(asv_pos)
        output = self.add([input, asv_pos])
        output = self.asv_dropout(output, training=training)
        for i in range(num_enc_layers):
            output = self.encoder_blocks[i](output, padding_mask=mask, training=training)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
                "dropout": self.dropout
        })
        return config

@tf.keras.saving.register_keras_serializable(package="amplicon_gpt", name="Memory")
class Memory(tf.keras.layers.Layer):
    def __init__(self, num_heads, mem_rows, mem_vec_size, **kwargs):
        """
        M -> nxm (mem_rows, mem_vec)
        K -> bxhxm (batch, head, mem_vec)
        E -> bxhxm (batch, head, mem_vec)
        w -> bxhxn
        """
        super().__init__(name="memory_head", **kwargs)
        self.mem_vec_size = mem_vec_size
        self.num_heads = num_heads
        self.mem_rows = mem_rows
        self.flatten = tf.keras.layers.Flatten()
        self.k_dropout = tf.keras.layers.Dropout(0.05)
        self.e_dropout = tf.keras.layers.Dropout(0.05)
        self.a_dropout = tf.keras.layers.Dropout(0.05)
        self.k = tf.keras.Sequential([
            tf.keras.layers.Dense(self.num_heads*self.mem_vec_size, activation=tf.keras.activations.linear),
            tf.keras.layers.Reshape((self.num_heads, self.mem_vec_size))
        ])
        # self.r = tf.keras.Sequential([
        #     tf.keras.layers.Dense(self.num_heads*self.mem_vec_size, kernel_initializer="ones", activation=tf.keras.activations.linear),
        #     tf.keras.layers.Reshape((self.num_heads, self.mem_vec_size))
        # ])
        self.e = tf.keras.Sequential([
            tf.keras.layers.Dense(self.num_heads*self.mem_vec_size, activation=tf.keras.activations.sigmoid),
            tf.keras.layers.Reshape((self.num_heads, self.mem_vec_size))
        ])
        self.a = tf.keras.Sequential([
            tf.keras.layers.Dense(self.num_heads*self.mem_vec_size, activation=tf.keras.activations.sigmoid),
            tf.keras.layers.Reshape((self.num_heads, self.mem_vec_size))
        ])
        
    
    def _compute_row_norm(self, x):
        return tf.norm(x, axis=-1, keepdims=True)
    
    def _compute_w(self, memory, inputs, training):
        """
        input -> x_t 
        k -> bxhxm
        M -> nxm (mem_rows, mem_vec
        """
        k = self.k(inputs)
        k_norm = k / self._compute_row_norm(k)
        k_norm = self.k_dropout(k_norm, training=training)

        M_norm = memory / self._compute_row_norm(memory)
        w = tf.einsum("...ij,...kj->...ik", k_norm, M_norm)
        w = tf.nn.softmax(w, axis=-1)
        return w
    
    def _erase_block(self, inputs, w, memory, training):
        e = self.e(inputs)
        e = self.e_dropout(e, training=training)
        erase = tf.einsum('bhn,bhmi->bnm', w, tf.expand_dims(e, axis=-1))
        erase_block = tf.math.multiply(memory, erase)
        return erase_block
    
    def _add_block(self, inputs, w, training):
        a = self.a(inputs)
        a = self.a_dropout(a, training=training)
        add = tf.einsum("bhn,bhmi->bnm", w, tf.expand_dims(a, axis=-1))
        return add
    
    def call(self, inputs, memory, w, training=False):
        old_w = self.flatten(w)
        inputs = tf.concat([old_w, inputs], axis=-1)
        w = self._compute_w(inputs, memory, training)
        erase_block = self._erase_block(inputs, w, memory, training)
        add_block = self._add_block(inputs, w, training)
        return memory - tf.multiply(memory, erase_block) + add_block, w

    def get_config(self):
        config = super().get_config()
        config.update({
            "mem_vec_size": self.mem_vec_size,
            "num_heads": self.num_heads,
            "mem_rows": self.mem_rows
        })
        return config
    
@tf.keras.saving.register_keras_serializable(package="amplicon_gpt", name="MemoryUnit")
class MemoryUnit(tf.keras.layers.Layer):
    def __init__(self, num_heads, mem_rows, mem_vec_size, **kwargs):
        super().__init__(self, name='memory_unit', **kwargs)
        self.num_heads = num_heads
        self.mem_rows = mem_rows
        self.mem_vec_size = mem_vec_size

        """
        k: amplify or attenuate the precision, tanh activation
        
        converts input dim BxIxM into BxHxM 
           weight matrix: IxH 
           I -> 2 + H (i.e. 1 for current, prev output, H-reads)
        """
        self.k_w = self.add_weight("k_w", (num_heads, 2 + num_heads))
        self.k_b = self.add_weight("k_bias", (1,num_heads))

        """
        beta: 1xH
        """
        self.beta = self.add_weight("beta", (mem_vec_size, ))
    
    def __feedforward(self, input, weight):
        return tf.einsum("...ij,...kj->...ik", input, weight)
    
    def call(self, inputs, memory, w, training=False):
        """
        This method is responsible for producing the 'emitting' vectors of each head
        input:  BxIxM
        """
        inputs = tf.transpose(inputs,perm=[0,2,1])
        
        k_vec = self.__feedforward(inputs, self.k_w) + self.k_b
        k_vec = tf.keras.activations.tanh(k_vec)
        k_vec = tf.transpose(k_vec,perm=[0,2,1])

        beta_vec = self.__feedforward(inputs, self.beta)
        beta_vec = tf.keras.activations.softplus(beta_vec)
        beta_vec = tf.transpose(beta_vec,perm=[0,2,1])

        # need to add a small delta in case this is 0
        k_norm = tf.norm(k_vec, axis=-1, keepdims=True) + 1e-6 
        k_vec = tf.divide(k_vec, k_norm)

        memory_norm = tf.norm(memory, axis=-1, keepdims=True) + 1e-6 
        memory = tf.divide(memory, memory_norm)

        memory = tf.transpose(memory, [0,2,1])
        kM = tf.einsum("bihm,bmn->bim", k_vec, memory)

        return kM    

@tf.keras.saving.register_keras_serializable(package="amplicon_gpt", name="ReadMemory")
class ReadMemory(tf.keras.layers.Layer):
    def __init__(self, num_heads, mem_rows, mem_vec_size):
        """
        M -> nxm (mem_rows, mem_vec)
        K -> bxhxm (batch, head, mem_vec)
        E -> bxhxm (batch, head, mem_vec)
        w -> bxhxn
        """
        super().__init__()
        self.r = tf.keras.Sequential([
            tf.keras.layers.Dense(num_heads*mem_vec_size, kernel_initializer="ones", activation=tf.keras.activations.linear),
            tf.keras.layers.Reshape((num_heads, mem_vec_size))
        ])
    
    def _compute_row_norm(self, x):
        return tf.norm(x, axis=-1, keepdims=True)

    def call(self, inputs, memory, w, training=False):
        old_w = self.flatten(w)
        inputs = tf.concat([old_w, inputs], axis=-1)
        r = self.r(inputs)
        r_norm = r / self._compute_row_norm(r)
        M_norm = memory / self._compute_row_norm(memory)
        w = tf.einsum("...ij,...kj->...ik", r_norm, M_norm)
        w = tf.nn.softmax(w, axis=-1)

        read = tf.einsum("bhn,bnm->bhm", w, memory)
        return read, w

    def get_config(self):
        config = super().get_config()
        config.update({
            "mem_vec_size": self.mem_vec_size,
            "num_heads": self.num_heads,
            "mem_rows": self.mem_rows
        })
        return config

@tf.keras.saving.register_keras_serializable(package="amplicon_gpt", name="NeuralMemory")
class NeuralMemory(tf.keras.layers.Layer):
    """
    last dimension needs to be the same going out and coming in
    """
    def __init__(self, num_heads, mem_rows, mem_vec_size, **kwargs):
        super().__init__(name="asv_embedding", **kwargs)
        self.num_heads = num_heads
        self.mem_vec_size = mem_vec_size
        self.mem_rows = mem_rows

        self.controller = tf.keras.layers.LSTM(32, return_state=True, return_sequences=False, dropout=0.5)
        self.memory = Memory(num_heads, mem_rows, mem_vec_size)
        self.reader = ReadMemory(num_heads, mem_rows, mem_vec_size)
        self.unit_init = UnitUniform(self.mem_rows)

    def build(self, input_shape):
        self.w_h = tf.Variable(lambda : self.unit_init((16, self.num_heads, self.mem_rows)),dtype=tf.float32)
        self.r_h = tf.Variable(lambda : self.unit_init((16, self.num_heads, self.mem_rows)),dtype=tf.float32)
        self.M = tf.Variable(tf.zeros((16, self.mem_rows, self.mem_vec_size)),dtype=tf.float32)

        self.prev_output = tf.Variable(tf.zeros((16, self.num_heads, self.mem_rows)),dtype=tf.float32)
        self.prev_reads = tf.Variable(tf.zeros((16, self.num_heads, self.mem_rows)),dtype=tf.float32)
        self.lstm_mem_state = tf.Variable(tf.zeros((16,32,)),dtype=tf.float32)
        self.lstm_cell_state = tf.Variable(tf.zeros((16,32)),dtype=tf.float32)

    def call(self, input, training=False):
        tf.print("1", input.shape)
        transposed_input = tf.transpose(input, perm=[1,0,2])
        tf.print("2", transposed_input.shape)
        time_steps = tf.shape(transposed_input)[0]

        def process_memory(inputs, time_steps):
            inputs = tf.expand_dims(inputs, axis=-2)
            memory_write = tf.TensorArray(tf.float32, 0, dynamic_size=True, clear_after_read=True)
            memory_write = memory_write.unstack(inputs)
            
            memory_read = tf.TensorArray(tf.float32, 0, dynamic_size=True, clear_after_read=True)
            memory_read = memory_read.unstack(inputs)

            for i in tf.range(0, time_steps):
                """
                Write memory: [input, prev out], k, prev output
                """
                write_input = tf.concat([memory_write.read(i), self.prev_output], axis=1)
                tf.print("3", write_input.shape)
                write_input = tf.concat([write_input, self.prev_reads], axis=1)
                tf.print("4", write_input.shape)
                new_memory, new_w_h = self.memory(write_input, self.w_h, self.M)
                self.w_h = self.w_h.assign(new_w_h)

                read_input = tf.concat([memory_read.read(i), self.prev_output], axis=1)
                read_input = tf.concat([read_input, self.prev_reads], axis=1)
                read, new_r_h = self.reader(read_input, self.r_h, self.M)
                self.r_h = self.r_h.assign(new_r_h)

                prev_output, lstm_mem_state, lstm_cell_state = self.controller(read, initial_state=[self.lstm_mem_state, self.lstm_cell_state])

                self.M = self.M.assign(new_memory)
                self.prev_reads = self.prev_reads.assign(read)
                self.prev_output = self.prev_output.assign(prev_output)
                self.lstm_mem_state = self.lstm_mem_state.assign(lstm_mem_state)
                self.lstm_cell_state = self.lstm_cell_state.assign(lstm_cell_state)
            
            return self.prev_output
        return process_memory(transposed_input, time_steps)

    def get_config(self):
        config = super().get_config()
        config.update({
            "mem_vec_size": self.mem_vec_size,
            "num_heads": self.num_heads,
            "mem_rows": self.mem_rows
        })
        return config
    