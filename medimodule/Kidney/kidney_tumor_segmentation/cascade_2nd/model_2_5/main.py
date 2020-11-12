# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np

from model import *
from load_data import *
from callback import *

# import keras
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

SEED = 42
np.random.seed(SEED)


def set_cbdir(root, model, task):
    for i in ['checkpoint', 'history', 'logs']:
        if not os.path.isdir(os.path.join(root, i)):
            os.mkdir(os.path.join(root, i))
        
        if not os.path.isdir(os.path.join(root, i, model)):
            os.mkdir(os.path.join(root, i, model))

        if not os.path.isdir(os.path.join(root, i, model, task)):
            os.mkdir(os.path.join(root, i, model, task))


def get_session():
    config = ConfigProto()
    config.gpu_options.allow_growth = True  # check needed
    return InteractiveSession(config=config)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", metavar="<command>", help="'train' or 'test'")
    parser.add_argument("--model", type=str, default="model_2", metavar="model_2/3/4/5")
    
    # data type
    parser.add_argument("--task", type=str, default="tumor", metavar="tumor / 1",
                        help="model_2 : tumor | model_3,4,5 : tumor1")

    # hyper-parameter
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lossfn", type=str, default="dice", metavar="dice / focaldice")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--wlevel", type=float, default=100.)
    parser.add_argument("--wwidth", type=float, default=400.)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--standard", type=str, default="minmax", metavar="minmax / normal")

    # etc
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--callback", type=int, default=0)
    parser.add_argument("--cb-root", type=str, default=None)
    
    return parser.parse_args()


def main():
    args = get_arguments()
    assert args.command in ["train", "test"]
    if args.model == 'model_2':
        assert args.task == 'tumor'
    else:
        assert args.task == 'tumor1'

    tf.keras.backend.tensorflow_backend.set_session(get_session())

    ##############################################
    # Set Hyper Parameters
    ##############################################
    MODEL = args.model
    TASK = args.task
    LR = args.lr
    LOSSFN = args.lossfn
    EPOCHS = args.epochs
    WLEVEL = args.wlevel
    WWIDTH = args.wwidth
    DEPTH = args.depth
    STANDARD = args.standard
    CB_ROOT = args.cb_root

    CLASSES = 3
    INPUT_SHAPE = (None, None, None, 1)

    CHECKPOINT = args.checkpoint
    INITIAL_EPOCH = 0

    trainset, testset = load()
    valset = trainset[:len(testset)]
    trainset = trainset[len(testset):]
    print('# of training data :', len(trainset), ', # of validation data :', len(valset))

    ##############################################
    # Set Model
    ##############################################
    model = MyModel(
        model=MODEL,
        input_shape=INPUT_SHAPE,
        lossfn=LOSSFN,
        classes=CLASSES,
        axis=-1,
        noise=0.1,
        depth=DEPTH,
        base_filter=32)

    if CHECKPOINT:
        model.mymodel.load_weights(CHECKPOINT)
        INITIAL_EPOCH = int(CHECKPOINT.split('_')[-2])
            
        print("Load weights successfully at {}".format(CHECKPOINT))
        print("Initial epoch :", INITIAL_EPOCH)

    model.compile(LR)

    if args.command == "train":
        ##############################################
        # Set Callbacks
        ##############################################
        if args.callback:
            set_cbdir(CB_ROOT, MODEL, TASK)
            callback_name = "{}_{}_{}_{}_{}_{}_{}".format(LR, LOSSFN, DIVIDE, WLEVEL, WWIDTH, DEPTH, STANDARD)

            cp = callback_checkpoint(filepath=os.path.join(CB_ROOT, 'checkpoint', MODEL, TASK, callback_name+'_{epoch:04d}_{val_tumor_dice:.4f}.h5'),
                                    monitor='val_tumor_dice',
                                    verbose=1,
                                    mode='max',
                                    save_best_only=True,
                                    save_weights_only=False)

            el = callback_epochlogger(filename=os.path.join(CB_ROOT, 'history', MODEL, TASK, callback_name+'_epoch.csv'),
                                        separator=',', append=True)

            tb = callback_tensorboard(log_dir=os.path.join(CB_ROOT, 'logs', MODEL, TASK, callback_name), batch_size=1)
            
            ls = callback_learningrate(LR, EPOCHS, INITIAL_EPOCH, TASK)
        
            callbacks = [cp, el, tb, ls]

        else:
            callbacks = []
        
        ##############################################
        # Train
        ##############################################
        train_generator = Generator(
            datalist=trainset,
            task=TASK,
            mode='train',
            standard=STANDARD,
            rotation_range=[10., 10., 10.] if MODEL == 'model_2' else [5., 5., 5.],
            wlevel=WLEVEL,
            wwidth=WWIDTH)

        val_generator = Generator(
            datalist=valset,
            task=TASK,
            mode='validation',
            standard=STANDARD,
            rotation_range=[0., 0., 0.],
            wlevel=WLEVEL,
            wwidth=WWIDTH,
            shuffle=False)

        total_generator = Generator(
            datalist=trainset+valset+testset,
            task=TASK,
            mode='train',
            standard=STANDARD,
            rotation_range=[10., 10., 10.] if MODEL == 'model_2' else [5., 5., 5.],
            wlevel=WLEVEL,
            wwidth=WWIDTH)

        steps_per_epoch = {'tumor': 332,
                           'tumor1': len(trainset)}
        validation_steps = {'tumor': 42,
                            'tumor1': len(valset)}
        total_steps = {'tumor': 332+42+42,
                       'tumor1': len(trainset+valset+testset)}

        # training for trainset
        model.mymodel.fit_generator(generator=train_generator,
                                    steps_per_epoch=steps_per_epoch[TASK],
                                    verbose=1,
                                    epochs=EPOCHS-20,
                                    validation_data=val_generator,
                                    validation_steps=validation_steps[TASK],
                                    callbacks=callbacks,
                                    shuffle=True,
                                    initial_epoch=INITIAL_EPOCH)

        # training for totalset without validation
        model.mymodel.fit_generator(generator=total_generator,
                                    steps_per_epoch=total_steps[TASK],
                                    verbose=1,
                                    epochs=EPOCHS,
                                    callbacks=callbacks,
                                    shuffle=True,
                                    initial_epoch=INITIAL_EPOCH)

    else:
        pass


if __name__ == "__main__":
    main()