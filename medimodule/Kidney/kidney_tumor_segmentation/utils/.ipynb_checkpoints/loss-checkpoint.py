# import keras
import tensorflow as tf
import tensorflow.keras.backend as K


def focal(alpha=0.25, gamma=2.0):
    def _focal(y_true, y_pred):
        loss = 0.
        label_length = y_pred.get_shape().as_list()[-1]
        y_pred = K.clip(y_pred, K.epsilon(), 1.-K.epsilon())

        if label_length > 1:
            for num_label in range(label_length):
                y_true_f = K.flatten(y_true[...,num_label])
                y_pred_f = K.flatten(y_pred[...,num_label])
                pt_1 = tf.where(tf.equal(y_true_f, 1), y_pred_f, tf.ones_like(y_pred_f))
                pt_0 = tf.where(tf.equal(y_true_f, 0), y_pred_f, tf.zeros_like(y_pred_f))
                loss += -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
            return loss / label_length
        else:
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            pt_1 = tf.where(tf.equal(y_true_f, 1), y_pred_f, tf.ones_like(y_pred_f))
            pt_0 = tf.where(tf.equal(y_true_f, 0), y_pred_f, tf.zeros_like(y_pred_f))
            loss += -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
            return loss
    return _focal


def dice_loss(y_true, y_pred, smooth=1.):
    loss = 0.
    label_length = y_pred.get_shape().as_list()[-1]
    y_pred = K.clip(y_pred, K.epsilon(), 1.-K.epsilon())

    if label_length > 1:
        for num_label in range(label_length):
            y_true_f = K.flatten(y_true[...,num_label])
            y_pred_f = K.flatten(y_pred[...,num_label])
            intersection = K.sum(y_true_f * y_pred_f)
            loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return 1-loss / label_length
    else:
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return 1-loss


def focal_dice_loss(y_true, y_pred):
    return focal(onehot)(y_true, y_pred) + dice_loss(onehot)(y_true, y_pred)