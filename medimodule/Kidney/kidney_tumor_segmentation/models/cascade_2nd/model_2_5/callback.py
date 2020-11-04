# import keras
import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import LearningRateScheduler


def callback_checkpoint(filepath, monitor, verbose, mode, save_best_only, save_weights_only):
    return ModelCheckpoint(filepath=filepath,
                           monitor=monitor,
                           verbose=verbose,
                           mode=mode,
                           save_best_only=save_best_only,
                           save_weights_only=save_weights_only)


def callback_epochlogger(filename, separator, append):
    return CSVLogger(filename=filename,
                     separator=separator,
                     append=append)


def callback_tensorboard(log_dir, batch_size):
    return TensorBoard(log_dir=log_dir, batch_size=batch_size)


def main_schedule(initlr, epochs, initial_epoch=0, task='tumor', warmup=5):
    def _cosine_anneal_schedule(e, te):
        cos_inner = np.pi * (e % te)
        cos_inner /= te
        cos_out = np.cos(cos_inner) + 1
        return float(initlr / 2 * cos_out)

    def _schedule(epoch, lr=None):
        if task == 'tumor1':
            if epoch > 29:
                return _cosine_anneal_schedule(epoch+1-29, epochs-29)
            else:
                return initlr
        else:
            if initial_epoch:
                return _cosine_anneal_schedule(epoch+1-initial_epoch, epochs-initial_epoch)
            else:
                if epoch+1 <= warmup:
                    return initlr * (epoch+1) / warmup
                else:
                    return _cosine_anneal_schedule(epoch+1, epochs)
            
    return _schedule


def callback_learningrate(initlr, epochs, initial_epoch=0, task='tumor'):
    return LearningRateScheduler(schedule=main_schedule(initlr, epochs+1, initial_epoch, task), verbose=1)