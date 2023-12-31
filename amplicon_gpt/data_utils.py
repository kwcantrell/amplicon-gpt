import numpy as np
import pandas as pd
import tensorflow as tf
from biom import load_table
from unifrac import unweighted

def create_base_sequencing_data(table_path, tree_path, batch_size, max_num_per_seq, seq_len, **kwargs):
    tree_path = tree_path
    table_path = table_path
    seq_len=seq_len
    table = load_table(table_path)
    randomize=False
    s_ids = table.ids(axis='sample')
    o_ids = np.array(table.ids(axis='observation'))
    samples = [np.argwhere(table.data(id, axis='sample') > 0).astype(np.int32).flatten() for id in s_ids]

    get_ids = lambda x: o_ids[x]
    get_seq = lambda x: [[*seq] for seq in x]
    equal = lambda x: [seq == np.array([[b'A'], [b'C'], [b'G'], [b'T']], dtype='<U1') for seq in x]
    argmax = lambda x: [np.argmax(seq, 0) + 1 for seq in x]
    sequencing_data = [np.apply_along_axis(
                        lambda sample: argmax(equal(get_seq(get_ids(sample)))), 0, sample)
    for sample in samples]
    unifrac_distances = unweighted(table_path, tree_path).data

    class Dataset:
        def __init__(self, sequencing_data, unifrac_distances, batch_size=16, num_epochs=10, randomize=False, items_per_epoch=None):
            self.sequencing_data = sequencing_data
            self.unifrac_distances = unifrac_distances
            self.batch_size = batch_size
            self.xs = np.arange(len(sequencing_data))
            self.num_epochs = num_epochs
            self.randomize = randomize
            if items_per_epoch is not None:
                self.items_per_epoch =  items_per_epoch
            else:
                self.items_per_epoch = int(len(unifrac_distances))
            self.on_epoch_end()

        def __len__(self):
            return int(len(self.sequencing_data)/self.batch_size)
        
        def __getitem__(self, index):
            start = index*self.batch_size
            pad = lambda x, pad_width: np.array([np.pad(z.tolist(),((0, pad_width - len(z)), (0,max_num_per_seq-seq_len))) for z in x])
            pad_width = lambda x: pad(x, np.max([len(z) for z in x]))
            sequence_batch = lambda i: pad_width([self.sequencing_data[i] for i in self.xs[i:i+self.batch_size]])
            unifrac_batch = lambda i: self.unifrac_distances[self.xs[i:i+self.batch_size], :][:, self.xs[i:i+self.batch_size]]
            get_batch = lambda i: (sequence_batch(i), unifrac_batch(i))
            return get_batch(start)
        
        def __call__(self):
            for _ in range(5):
                for i in range(self.__len__()):
                    yield self.__getitem__(i)
                self.on_epoch_end()

        def on_epoch_end(self):
            if self.randomize:
                tf.print('Randomizing data!!!!')
                np.random.shuffle(self.xs)

    dataset = Dataset(sequencing_data, unifrac_distances, batch_size, randomize=randomize)
    return tf.data.Dataset.from_generator(
        dataset,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, None, max_num_per_seq), dtype=tf.int32),
            tf.TensorSpec(shape=(batch_size, batch_size), dtype=tf.float32)
        )
    )

def create_veg_sequencing_data(table_path, batch_size, max_num_per_seq, seq_len, metadata_path, randomize=True, **kwargs):
    meta = pd.read_csv(metadata_path, sep='\t', index_col=0, dtype={'#SampleID':str})
    categories = np.array([1 if cat == 'high' else 0 for cat in meta['veg_cat']])
    categories = np.reshape(categories, (-1, 1))

    seq_len=seq_len
    table_path = table_path
    table = load_table(table_path)
    table.filter(meta.index, axis='sample')
    s_ids = table.ids(axis='sample')
    o_ids = np.array(table.ids(axis='observation'))
    samples = [np.argwhere(table.data(id, axis='sample') > 0).astype(np.int32).flatten() for id in s_ids]

    get_ids = lambda x: o_ids[x]
    get_seq = lambda x: [[*seq] for seq in x]
    equal = lambda x: [seq == np.array([[b'A'], [b'C'], [b'G'], [b'T']], dtype='<U1') for seq in x]
    argmax = lambda x: [np.argmax(seq, 0) + 1 for seq in x]
    sequencing_data = [np.apply_along_axis(
                        lambda sample: argmax(equal(get_seq(get_ids(sample)))), 0, sample)
    for sample in samples]

    
    return sequencing_data, categories

def create_veg_dataset(sequencing_data, categories, batch_size, randomize, limit_size, max_num_per_seq, seq_len, **kwargs):
    class Dataset:
        def __init__(self, sequencing_data, categories, batch_size=16, num_epochs=10, randomize=False, items_per_epoch=None):
            self.sequencing_data = sequencing_data
            self.categories = categories
            self.batch_size = batch_size
            self.xs = np.arange(len(sequencing_data))
            self.num_epochs = num_epochs
            self.randomize = randomize
            if items_per_epoch is not None:
                self.items_per_epoch =  items_per_epoch
            else:
                self.items_per_epoch = int(len(categories))
            self.on_epoch_end()
            self.limit_size = limit_size
            self.max_num_per_seq = max_num_per_seq
            self.seq_len = seq_len

        def __len__(self):
            return int((len(self.sequencing_data)*self.limit_size)/self.batch_size)
        
        def __getitem__(self, index):
            start = index*self.batch_size
            pad = lambda x, pad_width: np.array([np.pad(z.tolist(),((0, pad_width - len(z)), (0,self.max_num_per_seq-self.seq_len))) for z in x])
            pad_width = lambda x: pad(x, np.max([len(z) for z in x]))
            sequence_batch = lambda i: pad_width([self.sequencing_data[i] for i in self.xs[i:i+self.batch_size]])
            category_batch = lambda i: self.categories[self.xs[i:i+self.batch_size]]
            get_batch = lambda i: (sequence_batch(i), category_batch(i))
            return get_batch(start)
        
        def __call__(self):
            for i in range(self.__len__()):
                yield self.__getitem__(i)
            self.on_epoch_end()

        def on_epoch_end(self):
            if self.randomize:
                tf.print('Randomizing data!!!!')
                np.random.shuffle(self.xs)

    dataset = Dataset(sequencing_data, categories, batch_size, randomize=randomize)
    return tf.data.Dataset.from_generator(
        dataset,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, None, max_num_per_seq), dtype=tf.int32),
            tf.TensorSpec(shape=(batch_size, 1), dtype=tf.float32)
        )
    )
def _get_sequencing_data(table, metadata, group_step):
    # filter table to only include agp samples
    agp_meta = metadata
    agp_samples = agp_meta.index.to_list()
    table.filter(agp_samples, axis='sample', inplace=True)
    print('???', agp_meta.shape, table.shape)
    table.remove_empty()
    print('!!!', agp_meta.shape, table.shape)
    s_ids = table.ids(axis='sample')
    o_ids = np.array(table.ids(axis='observation'))
    samples = [np.argwhere(table.data(id, axis='sample') > 0.5).astype(np.int32).flatten() for id in s_ids]

    get_ids = lambda x: o_ids[x]
    get_seq = lambda x: [[*seq] for seq in x]
    equal = lambda x: [seq == np.array([[b'A'], [b'C'], [b'G'], [b'T']], dtype='<U1') for seq in x]
    argmax = lambda x: [np.argmax(seq, 0) + 1 for seq in x] # add one to account for mask token
    sequencing_data = [np.apply_along_axis(
                        lambda sample: argmax(equal(get_seq(get_ids(sample)))), 0, sample)
        for sample in samples]
    age_data = np.array(agp_meta['age'].tolist())

    step = group_step
    min_age = 25
    max_age = 100
    groups = []
    for i in range(min_age, max_age+1, step):
        print(agp_meta.loc[(agp_meta['age']  >= i-step)  & (agp_meta['age']  < i)].shape[0], i)
        groups.append(agp_meta.loc[(agp_meta['age']  >= i-step)  & (agp_meta['age']  < i)].shape[0])
    return sequencing_data, age_data

def create_unifrac_sequencing_data(table_path, tree_path, batch_size, max_num_per_seq, seq_len, repeat=1, split_percent=None, **kwargs):
    table = load_table(table_path)

    s_ids = table.ids(axis='sample')
    o_ids = np.array(table.ids(axis='observation'))

    samples = [np.argwhere(table.data(id, axis='sample') > 0).astype(np.int32).flatten() for id in s_ids]

    get_ids = lambda x: o_ids[x]
    get_seq = lambda x: [[*seq] for seq in x]
    equal = lambda x: [seq == np.array([[b'A'], [b'C'], [b'G'], [b'T']], dtype='<U1') for seq in x]
    argmax = lambda x: [np.argmax(seq, 0) + 1 for seq in x]
    sequencing_data = [np.apply_along_axis(
                        lambda sample: argmax(equal(get_seq(get_ids(sample)))), 0, sample)
    for sample in samples]
    unifrac_distances = unweighted(table_path, tree_path).data

    class Dataset:
        def __init__(self, sequencing_data, unifrac_distances, batch_size=32, randomize=False, repeat=1, max_num_per_seq=100, seq_len=100):
            self.sequencing_data = sequencing_data
            self.unifrac_distances = unifrac_distances
            self.batch_size = batch_size
            self.xs = np.arange(len(sequencing_data))
            self.randomize = randomize
            self.repeat = repeat
            self.max_num_per_seq = max_num_per_seq
            self.seq_len = seq_len
            self._shuffle()

        def __len__(self):
            return int(len(self.sequencing_data)/self.batch_size)
        
        def __getitem__(self, index):
            start = index*self.batch_size
            pad = lambda x, pad_width: np.array([np.pad(z.tolist(),((0, pad_width - len(z)), (0,self.max_num_per_seq-self.seq_len))) for z in x])
            pad_width = lambda x: pad(x, np.max([len(z) for z in x]))
            sequence_batch = lambda i: pad_width([self.sequencing_data[i] for i in self.xs[i:i+self.batch_size]])
            unifrac_batch = lambda i: self.unifrac_distances[self.xs[i:i+self.batch_size], :][:, self.xs[i:i+self.batch_size]]
            get_batch = lambda i: (sequence_batch(i), unifrac_batch(i))
            return get_batch(start)
        
        def __call__(self):
            for _ in range(self.repeat):
                for i in range(self.__len__()):
                    yield self.__getitem__(i)
                self._shuffle()

        def _shuffle(self):
            if self.randomize:
                np.random.shuffle(self.xs)

        def on_epoch_end(self):
            self._shuffle()
            

    if split_percent:
        s_id_indicies = np.arange(len(samples))
        np.random.shuffle(s_id_indicies)
        training_size = int(len(samples) * (1-split_percent))
        
        training_ids_ind = s_id_indicies[:training_size]
        training_seq = [sequencing_data[i] for i in training_ids_ind]
        training_dist = unifrac_distances[training_ids_ind, :][:, training_ids_ind]
        train_dataset = Dataset(training_seq, training_dist, batch_size=batch_size, randomize=True, repeat=repeat, max_num_per_seq=max_num_per_seq, seq_len=seq_len)

        val_ids_ind = s_id_indicies[training_size:]
        val_seq = [sequencing_data[i] for i in val_ids_ind]
        val_dist = unifrac_distances[val_ids_ind, :][:, val_ids_ind]
        val_dataset = Dataset(val_seq, val_dist, batch_size=batch_size, randomize=False, repeat=1, max_num_per_seq=max_num_per_seq, seq_len=seq_len)
        return tf.data.Dataset.from_generator(
            train_dataset,
            output_signature=(
                tf.TensorSpec(shape=(batch_size, None, max_num_per_seq), dtype=tf.int32),
                tf.TensorSpec(shape=(batch_size, batch_size), dtype=tf.float32)
            )
        ), tf.data.Dataset.from_generator(
            val_dataset,
            output_signature=(
                tf.TensorSpec(shape=(batch_size, None, max_num_per_seq), dtype=tf.int32),
                tf.TensorSpec(shape=(batch_size, batch_size), dtype=tf.float32)
            )
        )

    dataset = Dataset(sequencing_data, unifrac_distances, batch_size=batch_size, randomize=False, repeat=repeat)
    return tf.data.Dataset.from_generator(
        dataset,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, None, max_num_per_seq), dtype=tf.int32),
            tf.TensorSpec(shape=(batch_size, batch_size), dtype=tf.float32)
        )
    )

def create_sequencing_data(table_path, metadata_path, split_percent=None, group_step=25, **kwargs):
    """
    voc for embedding layer is <MASK> := 0, A := 1, C := 2, G := 3, T := 4
    """
    table = load_table(table_path)
    meta = pd.read_csv(metadata_path, sep='\t', index_col=0, dtype={'#SampleID':str})
    meta['age'] = meta['age'].astype(np.float32)

    if split_percent:
        training_df = meta.sample(frac=1-split_percent, replace=False, random_state=1)
        training_table = table.filter(training_df.index, axis='sample', inplace=False)
        validation_df = meta[~meta.index.isin(training_df.index)]
        validation_table = table.filter(validation_df.index, axis='sample', inplace=False)
        return (_get_sequencing_data(training_table, training_df, group_step),
                _get_sequencing_data(validation_table, validation_df, group_step)
        )
    else:
        return _get_sequencing_data(table, meta, group_step)
        
def create_dataset(sequencing_data, age_data, groups, batch_size, randomize, max_num_per_seq, seq_len, repeat=None, **kwargs):
    class Dataset:
        def __init__(self, sequencing_data, age_data, batch_size, groups, randomize, max_num_per_seq, seq_len, repeat=None, **kwargs):
            self.sequencing_data = sequencing_data
            self.age_data = age_data
            self.batch_size = batch_size
            if groups is None:
                self.xs = np.arange(len(sequencing_data))
            else:
                start = 0
                max_group_size = np.max(groups)
                self.sequence_groups = []
                self.num_groups = len(groups)
                for size in groups:
                    self.sequence_groups.append(np.tile(np.arange(start, start + size),
                                                        int(max_group_size / size) + 1))
                    start += size

            self.randomize=randomize
            self.max_num_per_seq = max_num_per_seq
            self.seq_len = seq_len
            if repeat is None:
                self.repeat = 1
            else:
                self.repeat = repeat
            self._shuffle()

        def __len__(self):

            return int(len(self.sequencing_data)/self.batch_size)
        
        def __getitem__(self, xs):
            pad = lambda x, pad_width: np.array([np.pad(z.tolist(),((0, pad_width - len(z)), (0,self.max_num_per_seq-self.seq_len))) for z in x])
            pad_width = lambda x: pad(x, np.max([len(z) for z in x]))
            sequence_batch = lambda xs: pad_width([self.sequencing_data[i] for i in xs])
            get_batch = lambda xs: (sequence_batch(xs), self.age_data[xs, np.newaxis])
            return get_batch(xs)
        
        def _get_xs(self, i):
            if hasattr(self, 'xs'):
                start = i*self.batch_size
                xs = self.xs[start:start+self.batch_size]
            else:
                start = i*int(self.batch_size/self.num_groups)
                end = start + int(self.batch_size/self.num_groups)
                xs = self.sequence_groups[0][start:end]
                for group_inds in self.sequence_groups[1:]:
                    xs = np.append(xs, group_inds[start:end])
            return xs
        
        def __call__(self):
            for j in range(self.repeat):
                for i in range(self.__len__()):
                    xs = self._get_xs(i)
                    yield self.__getitem__(xs)
                self._shuffle()

        def _shuffle(self):
            if self.randomize:
                if hasattr(self, 'xs'):
                    np.random.shuffle(self.xs)
                else:
                    for group_inds in self.sequence_groups:
                        np.random.shuffle(group_inds)
        def on_epoch_end(self):
            self._shuffle()

    from_generator = lambda dataset: tf.data.Dataset.from_generator(
        dataset,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, None, 150), dtype=tf.int32),
            tf.TensorSpec(shape=(batch_size, 1), dtype=tf.float32)
    ))
    dataset = Dataset(sequencing_data, age_data, batch_size, groups, randomize, max_num_per_seq, seq_len, repeat)
    return from_generator(dataset)


