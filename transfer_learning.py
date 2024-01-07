import click
import os
import json
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import amplicon_gpt._parameter_descriptions as desc
from amplicon_gpt.callbacks import MAE_Scatter, mean_absolute_error, mean_confidence_interval, Accuracy, ProjectEncoder
from amplicon_gpt.data_utils import (
    create_sequencing_data, create_dataset, create_veg_sequencing_data, create_veg_dataset, create_unifrac_sequencing_data,
    get_sequencing_dataset, get_unifrac_dataset, combine_seq_dist_dataset, batch_dist_dataset
)
from amplicon_gpt.model_utils import transfer_learn_feature_regression, transfer_learn_feature_classification, transfer_learn_base, compile_model

# Allow using -h to show help information
# https://click.palletsprojects.com/en/7.x/documentation/#help-parameter-customization
CTXSETS = {"help_option_names": ["-h", "--help"]}

@click.group()
def transfer_learning():
    pass

@transfer_learning.command(
        'regression', 
        short_help=desc.REGRESSION,
        context_settings=CTXSETS
)
@click.option(
    '--config-json',
    required=True,
    type=click.Path(exists=True),
    help=desc.CONFIG_JSON
)
@click.option(
    '-c', '--continue-training',
    required=False, default=False, is_flag=True,
    help=desc.CONTINUE_TRAINING, 
)
@click.option(
    '--output-model-summary',
    required=False, default=False, is_flag=True,
    help=desc.OUTPUT_MODEL_SUMMARY
)
def regression(config_json, continue_training, output_model_summary):
    with open(config_json) as f:
        config = json.load(f)

    initial_epoch=0
    if continue_training:
        initial_epoch=0 # load finishing epoch from root directory

    (training_seq, training_age), (val_seq, val_age) = create_sequencing_data(split_percent=config['validation_percent'], **config)
    training_dataset = create_dataset(training_seq, training_age, groups=None, randomize=True, repeat=config['mini_epochs'], **config)
    validation_dataset = create_dataset(val_seq, val_age, groups=None, randomize=False, **config)
    model = transfer_learn_feature_regression(continue_training, **config)

    if output_model_summary:
        model.summary()
    
    if 'patience' in config:
        patience=config['patience']
    else:
        patience=10

    t_dataset = create_dataset(training_seq, training_age, groups=None, randomize=False, **config)
    
    # model.fit(X_train, Y_train, callbacks=[reduce_lr])

    model.fit(
        training_dataset, validation_data=validation_dataset,
         epochs=config['epochs'], initial_epoch=0, batch_size=config['batch_size'],
         callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', start_from_epoch=0, patience=patience, mode='min'),
                    MAE_Scatter(title='training', dataset=t_dataset, **config),
                    MAE_Scatter(title='validation', dataset=validation_dataset, **config)]
    )
    model.save(os.path.join(config['root_path'], 'model.keras'), save_format='keras')

@transfer_learning.command('unifrac')
@click.option(
    '--config-json',
    required=True,
    type=click.Path(exists=True),
    help=desc.CONFIG_JSON
)
@click.option(
    '-c', '--continue-training',
    required=False, default=False, is_flag=True,
    help=desc.CONTINUE_TRAINING, 
)
@click.option(
    '--output-model-summary',
    required=False, default=False, is_flag=True,
    help=desc.OUTPUT_MODEL_SUMMARY
)
def unifrac(config_json, continue_training, output_model_summary):
    with open(config_json) as f:
        config = json.load(f)

    seq_dataset = get_sequencing_dataset(**config)
    unifrac_dataset = get_unifrac_dataset(**config)
    sequence_tokenizer = tf.keras.layers.TextVectorization(max_tokens=7, split='character', output_mode='int', output_sequence_length=100)
    sequence_tokenizer.adapt(seq_dataset.take(1))
    dataset = combine_seq_dist_dataset(seq_dataset, unifrac_dataset, **config)

    size = seq_dataset.cardinality().numpy()
    batch_size = config['batch_size']
    train_size = int(size*config['train_percent']/batch_size)*batch_size

    training_dataset = dataset.take(train_size).prefetch(tf.data.AUTOTUNE)
    training_dataset = batch_dist_dataset(training_dataset, shuffle=True, **config)
    
    val_data = dataset.skip(train_size).prefetch(tf.data.AUTOTUNE)
    validation_dataset = batch_dist_dataset(val_data, **config)

    model = transfer_learn_base(sequence_tokenizer=sequence_tokenizer, load_prev_path=False, **config)
    
    model = compile_model(model)
    for x, _ in training_dataset.take(1):
        y = model(x)
    
    if output_model_summary:
        model.summary()
    
    if 'patience' in config:
        patience=config['patience']
    else:
        patience=10
    config['repeat'] = 1
    
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                           patience=5, min_lr=0.001)
    
    model.fit(
        training_dataset, validation_data=validation_dataset,
        epochs=config['epochs'], initial_epoch=0, batch_size=config['batch_size'],
        callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', start_from_epoch=0, patience=patience, mode='min'),
                    ProjectEncoder(seq_dataset.padded_batch(config['batch_size']),**config)
        ]
    )
    model.save(os.path.join(config['root_path'], 'model.keras'), save_format='keras')

@transfer_learning.command('veg_classifier')
@click.pass_context
@click.option('--config-json', type=click.Path(exists=True))
def veg_classifier(ctx, config_json):
    with open(config_json) as f:
        config = json.load(f)
    training_percent = config['training_percent']
    validation_percent = config['validation_percent']
    acc_percent = config['acc_percent']
    batch_size = config['batch_size']
    epochs = config['epochs']

    sequencing_data, categories = create_veg_sequencing_data(**config)
    training_dataset = create_veg_dataset(sequencing_data, categories, randomize=True, limit_size=training_percent, **config)
    validation_dataset = create_veg_dataset(sequencing_data, categories, randomize=True, limit_size=validation_percent, **config)
    model = transfer_learn_feature_classification(**config)
    model.summary()

    acc_dataset = create_veg_dataset(sequencing_data, categories, randomize=False, limit_size=.5, **config)
    if 'patience' in config:
        patience=config['patience']
    else:
        patience=10
    model.fit(
        training_dataset, validation_data=validation_dataset,
         epochs=epochs, initial_epoch=0, batch_size=batch_size,
         callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', start_from_epoch=0, patience=patience, mode='max'),
                    Accuracy(dataset=acc_dataset, **config)]
    )
    model.save(os.path.join(config['root_path'], 'model.keras'), save_format='keras')

@transfer_learning.command('veg_auc')
@click.pass_context
@click.option('--config-json', type=click.Path(exists=True))
def veg_auc(ctx, config_json):
    with open(config_json) as f:
        config = json.load(f)
    acc_percent = config['acc_percent']
    sequencing_data, categories = create_veg_sequencing_data(**config)
    model = transfer_learn_feature_classification(**config)
    model.summary()
    acc_dataset = create_veg_dataset(sequencing_data, categories, randomize=False, limit_size=acc_percent, **config)
    fname=os.path.join('agp/veg-cat', 'auc.png')
    import sklearn
    import matplotlib.pyplot as plt
    def plot_prc(name, labels, predictions, **kwargs):
        precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)

        plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.grid(True)
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.savefig(fname)

    def plot_roc(name, labels, predictions, **kwargs):
        fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

        plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
        plt.xlabel('False positives [%]')
        plt.ylabel('True positives [%]')
        plt.xlim([-0.5,20])
        plt.ylim([80,100.5])
        plt.grid(True)
        ax = plt.gca()
        ax.set_aspect('equal')
        fname=os.path.join('agp/veg-cat', 'roc.png')
        plt.savefig(fname)


    pred_cat = tf.squeeze(model.predict(acc_dataset)).numpy()
    true_cat = np.concatenate([tf.squeeze(ys).numpy() for (_, ys) in acc_dataset])
    plot_prc('AUC', true_cat, pred_cat)
    plot_roc('ROC', true_cat, pred_cat)


@transfer_learning.command('mae_plot')
@click.pass_context
@click.option('--config-json', type=click.Path(exists=True))
def mae_plot(ctx, config_json):
    with open(config_json) as f:
        config = json.load(f)
    config['load_prev_path'] = True
    sequencing_data, age_data, groups = create_sequencing_data(**config)
    model = transfer_learn_feature_regression(**config)
    mae_dataset = create_dataset(sequencing_data, age_data, groups=None, randomize=False, limit_size=1.0, **config)
    mean_absolute_error(mae_dataset, model, config['final_figure_path'], config['s_type'])

def main():
    transfer_learning(prog_name='transfer_learning')

if __name__ == '__main__':
    main()