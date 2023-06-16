import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam, SGD, Adadelta, Nadam
from tensorflow_addons.optimizers import SGDW, AdamW, AdaBelief
from tensorflow_addons.optimizers import CyclicalLearningRate
from keras import mixed_precision
from utils.models import Unet, Residual_Unet, Attention_Unet, Unet_plus, DeepLabV3plus
from utils.losses import IoULoss, DiceLoss, TverskyLoss, FocalTverskyLoss, HybridLoss, FocalHybridLoss
from utils.datasets import CityscapesDataset, MapillaryDataset
from utils.eval import MeanIoU
from utils import utils
from argparse import ArgumentParser
import yaml

parser = ArgumentParser('')
parser.add_argument('--config', type=str, nargs='?', default='config/cityscapes.yaml')
parser.add_argument('--data_path', type=str, nargs='?')
parser.add_argument('--dataset', type=str, nargs='?', default='Cityscapes', choices=['Cityscapes', 'Mapillary'])
parser.add_argument('--model_type', type=str, nargs='?', choices=['Unet', 'Residual_Unet', 'Attention_Unet', 'Unet_plus', 'DeepLabV3plus'])
parser.add_argument('--model_name', type=str, nargs='?')
parser.add_argument('--backbone', type=str, nargs='?', default='None')
parser.add_argument('--output_stride', type=int, nargs='?', default=32)
parser.add_argument('--unfreeze_at', type=str, nargs='?')
parser.add_argument('--activation', type=str, nargs='?', default='relu')
parser.add_argument('--dropout', type=float, nargs='?', default=0.0)
parser.add_argument('--optimizer', type=str, nargs='?', default='Adam', choices=['Adam', 'Adadelta', 'Nadam', 'AdaBelief', 'AdamW', 'SGDW'])
parser.add_argument('--loss', type=str, nargs='?', default='FocalHybridLoss', choices=['DiceLoss', 'IoULoss', 'TverskyLoss', 'FocalTverskyLoss', 'HybridLoss', 'FocalHybridLoss'])
parser.add_argument('--batch_size', type=int, nargs='?', default='3')
parser.add_argument('--augment', type=bool, nargs='?', default=False)
parser.add_argument('--epochs', type=int, nargs='?', default='20')
parser.add_argument('--final_epochs', type=int, nargs='?', default='60')
args = parser.parse_args()


if args.config is None:
    # parse arguments
    print('Reading configuration from cmd args')
    DATA_PATH = args.data_path
    DATASET = args.dataset
    MODEL_TYPE = args.model_type
    MODEL_NAME = args.model_name
    BACKBONE = args.backbone
    OUTPUT_STRIDE = args.output_stride
    OPTIMIZER_NAME = args.optimizer
    UNFREEZE_AT = args.unfreeze_at
    LOSS = args.loss
    BATCH_SIZE = args.batch_size
    ACTIVATION = args.activation
    DROPOUT_RATE = args.dropout
    AUGMENT = args.augment
    EPOCHS = args.epochs
    FINAL_EPOCHS = args.final_epochs
    
else:
    # Read YAML file
    print('Reading configuration from config yaml')
    
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    LOGS_DIR = config['logs_dir']
    model_config = config['model']
    dataset_config = config['dataset']
    train_config = config['train_config']

    # Dataset Configuration
    DATASET = dataset_config['name']
    DATA_PATH = dataset_config['path']
    VERSION = dataset_config['version']
    NUM_TRAIN_IMAGES = dataset_config['num_train_images']
    NUM_EVAL_IMAGES = dataset_config['num_eval_images']
    CACHE = dataset_config['cache']
    CACHE_FILE = dataset_config['cache_file']
    SEED = dataset_config['seed']
    INPUT_TRAINING = dataset_config['input_training']
    LABEL_TRAINING = dataset_config['label_training']
    INPUT_VALIDATION = dataset_config['input_validation']
    LABEL_VALIDATION = dataset_config['label_validation']
    MAX_SAMPLES_TRAINING = dataset_config['max_samples_training']
    MAX_SAMPLES_VALIDATION = dataset_config['max_samples_validation']
    ONE_HOT_PALETTE_LABEL = dataset_config['one_hot_palette_label']

    # Model Configuration
    MODEL_TYPE = model_config['architecture']
    MODEL_NAME = model_config['name']
    BACKBONE = model_config['backbone']
    UNFREEZE_AT = model_config['unfreeze_at']
    INPUT_SHAPE = model_config['input_shape']
    OUTPUT_STRIDE = model_config['output_stride']
    FILTERS = model_config['filters']
    ACTIVATION = model_config['activation']
    DROPOUT_RATE = model_config['dropout_rate']

    # Training Configuration
    PRETRAINED_WEIGHTS = model_config['pretrained_weights']
    
    BATCH_SIZE = train_config['batch_size']
    EPOCHS = train_config['epochs']
    FINAL_EPOCHS = train_config['final_epochs']
    AUGMENT = train_config['augment']
    MIXED_PRECISION = train_config['mixed_precision']
    LOSS = train_config['loss']

    optimizer_config = train_config['optimizer']
    OPTIMIZER_NAME = optimizer_config['name']
    WEIGHT_DECAY = optimizer_config['weight_decay']
    MOMENTUM = optimizer_config['momentum']
    START_LR = optimizer_config['schedule']['start_lr']
    END_LR = optimizer_config['schedule']['end_lr']
    LR_DECAY_EPOCHS = optimizer_config['schedule']['decay_epochs']
    POWER = optimizer_config['schedule']['power']

    DISTRIBUTE_STRATEGY = train_config['distribute']['strategy']
    DEVICES = train_config['distribute']['devices']

if DATASET == 'Cityscapes':
    NUM_CLASSES = 20
    IGNORE_CLASS = 19
    INPUT_SHAPE = INPUT_SHAPE
elif DATASET == 'Mapillary':
    INPUT_SHAPE = (1024, 1856, 3)
    if VERSION == 'v1.2':
        NUM_CLASSES = 64
        IGNORE_CLASS = 63
    elif VERSION == 'v2.0':
        NUM_CLASSES = 118
        IGNORE_CLASS = 117
    else:
        raise ValueError('Version of the Mapillary Vistas dataset should be either v1.2 or v2.0!')
else:
    raise ValueError(F'{DATASET} dataset is invalid. Available Datasets are: Cityscapes, Mapillary!')

# Define preprocessing according to the Backbone
if BACKBONE == 'None':
    PREPROCESSING = 'default'
    BACKBONE = None
elif 'ResNet' in BACKBONE:
    PREPROCESSING = 'ResNet'
    if 'V2' in BACKBONE:
        PREPROCESSING = 'ResNetV2'
elif 'EfficientNet' in BACKBONE:
    PREPROCESSING = 'EfficientNet'
elif 'EfficientNetV2' in BACKBONE:
    PREPROCESSING = 'EfficientNetV2'
elif 'MobileNet' == BACKBONE:
    PREPROCESSING = 'MobileNet'
elif 'MobileNetV2' == BACKBONE:
    PREPROCESSING = 'MobileNetV2'
elif 'MobileNetV3' in BACKBONE:
    PREPROCESSING = 'MobileNetV3'
elif 'RegNet' in BACKBONE:
    PREPROCESSING = 'RegNet'
else:
    raise ValueError(f'Enter a valid Backbone name, {BACKBONE} is invalid.')

if MIXED_PRECISION:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

# --------------------------------------------------------------------------------
# determine absolute filepaths
INPUT_TRAINING = utils.abspath(INPUT_TRAINING)
LABEL_TRAINING = utils.abspath(LABEL_TRAINING)
INPUT_VALIDATION = utils.abspath(INPUT_VALIDATION)
LABEL_VALIDATION = utils.abspath(LABEL_VALIDATION)

# get image and label file names for training and validation
# get max_samples_training random training samples
# TODO: consider images and labels when there names matches
files_train_input = utils.get_files_recursive(INPUT_TRAINING)
files_train_label = utils.get_files_recursive(LABEL_TRAINING, "color")
_, idcs = utils.sample_list(files_train_label, n_samples=MAX_SAMPLES_TRAINING)
files_train_input = np.take(files_train_input, idcs)
files_train_label = np.take(files_train_label, idcs)
image_shape_original_input = utils.load_image(files_train_input[0]).shape[0:2]
image_shape_original_label = utils.load_image(files_train_label[0]).shape[0:2]
print(f"Found {len(files_train_label)} training samples")

# get max_samples_validation random validation samples
files_valid_input = utils.get_files_recursive(INPUT_VALIDATION)
files_valid_label = utils.get_files_recursive(LABEL_VALIDATION, "color")
_, idcs = utils.sample_list(files_valid_label, n_samples=MAX_SAMPLES_VALIDATION)
files_valid_input = np.take(files_valid_input, idcs)
files_valid_label = np.take(files_valid_label, idcs)
print(f"Found {len(files_valid_label)} validation samples")

# parse one-hot-conversion.xml
_, ONE_HOT_PALETTE_LABEL = utils.parse_convert_py(ONE_HOT_PALETTE_LABEL)
assert NUM_CLASSES == len(ONE_HOT_PALETTE_LABEL)

# data augmentation setting

# data generator
# build dataset pipeline parsing functions
def parse_sample(input_files, label_file):
    # parse and process input images
    input = utils.load_image_op(input_files)
    input = utils.resize_image_op(input, image_shape_original_input, INPUT_SHAPE[0:2], interpolation=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # normalise the image
    input = utils.normalise_image_op(input)
    
    # parse and process label image
    label = utils.load_image_op(label_file)
    label = utils.resize_image_op(label, image_shape_original_label, INPUT_SHAPE[0:2], interpolation=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # one hot encode the label
    # label = utils.one_hot_encode_gray_op(label, conf.num_classes)
    label = utils.one_hot_encode_label_op(label, ONE_HOT_PALETTE_LABEL)
    return input, label

# build training data pipeline
dataTrain = tf.data.Dataset.from_tensor_slices((files_train_input, files_train_label))
dataTrain = dataTrain.shuffle(buffer_size=MAX_SAMPLES_TRAINING, reshuffle_each_iteration=True)
dataTrain = dataTrain.map(parse_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataTrain = dataTrain.batch(BATCH_SIZE, drop_remainder=True)
dataTrain = dataTrain.repeat(EPOCHS)
dataTrain = dataTrain.prefetch(1)
print("Built data pipeline for training")

# build validation data pipeline
dataValid = tf.data.Dataset.from_tensor_slices((files_valid_input, files_valid_label))
dataValid = dataValid.map(parse_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataValid = dataValid.batch(BATCH_SIZE, drop_remainder=True)
dataValid = dataValid.repeat(EPOCHS)
dataValid = dataValid.prefetch(1)
print("Built data pipeline for validation")

n_batches_train = dataTrain.cardinality().numpy() // EPOCHS
n_batches_valid = dataValid.cardinality().numpy() // EPOCHS

# ---------------------------Create Dataset stream--------------------------------
# if DATASET == 'Cityscapes':
#     train_ds = CityscapesDataset(num_classes=NUM_CLASSES, 
#                                  split='train', 
#                                  preprocessing=PREPROCESSING, 
#                                  shuffle=True, 
#                                  cache=CACHE,
#                                  cache_file=CACHE_FILE
#                                  )
#     train_ds = train_ds.create(DATA_PATH, 'all', BATCH_SIZE, NUM_TRAIN_IMAGES, augment=False, seed=SEED)

#     val_ds = CityscapesDataset(num_classes=NUM_CLASSES, 
#                                split='val', 
#                                preprocessing=PREPROCESSING, 
#                                shuffle=False,
#                                cache=CACHE,
#                                cache_file=CACHE_FILE
#                                )
#     val_ds = val_ds.create(DATA_PATH, 'all', BATCH_SIZE, NUM_EVAL_IMAGES, seed=SEED)
    
# elif DATASET == 'Mapillary':
#     train_ds = MapillaryDataset(height=1024, width=1856,
#                                 split='training',
#                                 preprocessing=PREPROCESSING,
#                                 version=VERSION,
#                                 shuffle=True,
#                                 )
#     train_ds = train_ds.create(DATA_PATH, BATCH_SIZE, NUM_TRAIN_IMAGES, augment=False, seed=SEED)

#     val_ds = MapillaryDataset(height=1024, width=1856,
#                               split='validation',
#                               preprocessing=PREPROCESSING,
#                               version=VERSION,
#                               shuffle=False)
#     val_ds = val_ds.create(DATA_PATH, BATCH_SIZE, NUM_EVAL_IMAGES, seed=SEED)

# steps_per_epoch = train_ds.cardinality().numpy()

# ---------------------------------------CALLBACKS--------------------------------
if BACKBONE is None:
    save_best_only = True
    save_freq = 'epoch'
else:
    save_best_only = False
    save_freq = int(EPOCHS*n_batches_train) # save the model only at the last epoch of the main training

checkpoint_filepath = f'saved_models/{DATASET}/{MODEL_TYPE}/{MODEL_NAME}'
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,                                           
                                            save_weights_only=False,
                                            monitor='val_MeanIoU_ignore',
                                            mode='max',
                                            save_freq=save_freq, 
                                            save_best_only=save_best_only,
                                            verbose=0)
#{LOGS_DIR}/
tensorboard_log_dir = f'Tensorboard_logs/{DATASET}/{MODEL_TYPE}/{MODEL_NAME}'
tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir,
                                   histogram_freq=0,
                                   write_graph=False,
                                   write_steps_per_second=False)

callbacks = [model_checkpoint_callback, tensorboard_callback]
# -------------------------------------------------------------------------------------------

loss_func = eval(LOSS)
loss = loss_func()

lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=START_LR,
    decay_steps=LR_DECAY_EPOCHS*n_batches_train,
    end_learning_rate=END_LR,
    power=POWER,
    cycle=False,
    name=None
    )

optimizer_dict = {
    'Adam' : Adam(lr_schedule),
    'Adadelta' : Adadelta(lr_schedule),
    'AdamW' : AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY),
    'AdaBelief' : AdaBelief(learning_rate=lr_schedule),
    'SGDW' : SGDW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
}

optimizer = optimizer_dict[OPTIMIZER_NAME]

mean_iou = MeanIoU(NUM_CLASSES, name='MeanIoU', ignore_class=None)
mean_iou_ignore = MeanIoU(NUM_CLASSES, name='MeanIoU_ignore', ignore_class=IGNORE_CLASS)
metrics = [mean_iou_ignore]

# Instantiate Model
model_function = eval(MODEL_TYPE)
model = model_function(input_shape=INPUT_SHAPE,
                       filters=FILTERS,
                       num_classes=NUM_CLASSES,
                       output_stride=OUTPUT_STRIDE,
                       activation=ACTIVATION,
                       dropout_rate=DROPOUT_RATE,
                       backbone_name=BACKBONE,
                       freeze_backbone=True,
                       weights=PRETRAINED_WEIGHTS
                       )
model.summary()

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

history = model.fit(dataTrain, steps_per_epoch=n_batches_train,
                    validation_data=dataValid, validation_freq=1, validation_steps=n_batches_valid,
                    epochs=EPOCHS,
                    callbacks=callbacks,
                    verbose=1
                    )

# FINE TUNE MODEL
if BACKBONE is not None:
    #* After unfreezing the final backbone weights the batch size might need to be reduced to
    #* prevent OOM. Re-define the dataset streams with new batch size
    
    # if DATASET == 'Cityscapes':
    #     train_ds = CityscapesDataset(num_classes=NUM_CLASSES, 
    #                                 split='train', 
    #                                 preprocessing=PREPROCESSING, 
    #                                 shuffle=True, 
    #                                 cache=CACHE,
    #                                 cache_file=CACHE_FILE
    #                                 )
    #     train_ds = train_ds.create(DATA_PATH, 'all', BATCH_SIZE-1, NUM_TRAIN_IMAGES, augment=AUGMENT, seed=SEED)

    #     val_ds = CityscapesDataset(num_classes=NUM_CLASSES, 
    #                             split='val', 
    #                             preprocessing=PREPROCESSING, 
    #                             shuffle=False,
    #                             cache=CACHE,
    #                             cache_file=CACHE_FILE
    #                             )
    #     val_ds = val_ds.create(DATA_PATH, 'all', BATCH_SIZE-1, NUM_EVAL_IMAGES, seed=SEED)
        
    # elif DATASET == 'Mapillary':
    #     train_ds = MapillaryDataset(height=1024, width=1856,
    #                                 split='training',
    #                                 preprocessing=PREPROCESSING,
    #                                 version=VERSION,
    #                                 shuffle=True,
    #                                 )
    #     train_ds = train_ds.create(DATA_PATH, BATCH_SIZE-1, NUM_TRAIN_IMAGES, augment=AUGMENT, seed=SEED)

    #     val_ds = MapillaryDataset(height=1024, width=1856,
    #                             split='validation',
    #                             preprocessing=PREPROCESSING,
    #                             version=VERSION,
    #                             shuffle=False)
    #     val_ds = val_ds.create(DATA_PATH, BATCH_SIZE-1, NUM_EVAL_IMAGES, seed=SEED)
    
    # build training data pipeline
    dataTrain = tf.data.Dataset.from_tensor_slices((files_train_input, files_train_label))
    dataTrain = dataTrain.shuffle(buffer_size=MAX_SAMPLES_TRAINING, reshuffle_each_iteration=True)
    dataTrain = dataTrain.map(parse_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataTrain = dataTrain.batch(BATCH_SIZE-2, drop_remainder=True)
    dataTrain = dataTrain.repeat(EPOCHS)
    dataTrain = dataTrain.prefetch(1)
    print("Built data pipeline for training")

    # build validation data pipeline
    dataValid = tf.data.Dataset.from_tensor_slices((files_valid_input, files_valid_label))
    dataValid = dataValid.map(parse_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataValid = dataValid.batch(BATCH_SIZE-2, drop_remainder=True)
    dataValid = dataValid.repeat(EPOCHS)
    dataValid = dataValid.prefetch(1)
    print("Built data pipeline for validation")

    n_batches_train = dataTrain.cardinality().numpy() // EPOCHS
    n_batches_valid = dataValid.cardinality().numpy() // EPOCHS

    # Re-define checkpoint callback to save only the best model
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,                                           
                                                save_weights_only=False,
                                                monitor='val_MeanIoU_ignore',
                                                mode='max',
                                                save_best_only=True,
                                                verbose=0)
    
    callbacks = [model_checkpoint_callback, tensorboard_callback]
    
    # instantiate model again with the last part of the encoder (Backbone) un-frozen
    model = model_function(input_shape=INPUT_SHAPE,
                           filters=FILTERS,
                           num_classes=NUM_CLASSES,
                           output_stride=OUTPUT_STRIDE,
                           activation=ACTIVATION,
                           dropout_rate=DROPOUT_RATE,
                           backbone_name=BACKBONE,
                           freeze_backbone=False,
                           unfreeze_at=UNFREEZE_AT,
                           )
    
    # load the saved weights into the model to fine tune the high level features of the feature extractor
    # Fine tune the encoder network with a lower learning rate
    model.load_weights(checkpoint_filepath)
    
    model.summary()
    
    optimizer_dict = {
    'Adam' : Adam(END_LR),
    'Adadelta' : Adadelta(END_LR),
    'AdamW' : AdamW(learning_rate=END_LR, weight_decay=WEIGHT_DECAY),
    'AdaBelief' : AdaBelief(learning_rate=END_LR, weight_decay=WEIGHT_DECAY),
    'SGDW' : SGDW(learning_rate=END_LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    }

    optimizer = optimizer_dict[OPTIMIZER_NAME]
    
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    history = model.fit(dataTrain, steps_per_epoch=n_batches_train,
                        validation_data=dataValid, validation_freq=1, validation_steps=n_batches_valid,
                        initial_epoch=EPOCHS,
                        epochs=FINAL_EPOCHS,
                        callbacks=callbacks,
                        verbose=1
                        )
    
    # TODO: write callback to save model trunk to avoid the following 
    if DATASET == 'Mapillary':
        model.save_weights(f'pretrained_models/{MODEL_TYPE}/{MODEL_NAME}/model')
        trunk = model.get_layer('Trunk')
        trunk.save_weights(f'pretrained_models/{MODEL_TYPE}/{MODEL_NAME}/trunk')