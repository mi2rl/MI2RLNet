# import keras
import tensorflow.keras.backend as K


def dice(y_true, y_pred, classes=None, smooth=1.):
    loss = 0.
    label_length = classes if classes else y_pred.get_shape().as_list()[-1]
    y_true = K.cast(y_true > 0.5, dtype=y_true.dtype)
    y_pred = K.cast(y_pred > 0.5, dtype=y_pred.dtype)

    if label_length > 1:
        for num_label in range(label_length):
            y_true_f = K.flatten(y_true[...,num_label])
            y_pred_f = K.flatten(y_pred[...,num_label])
            intersection = K.sum(y_true_f * y_pred_f)
            loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return loss / label_length
    else:
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return loss


def kidney_dice(y_true, y_pred):
    return dice(y_true[...,1], y_pred[...,1], classes=1)


def tumor_dice(y_true, y_pred):
    return dice(y_true[...,2], y_pred[...,2], classes=1)


def iou(y_true, y_pred):
    loss = 0.
    label_length = y_pred.get_shape().as_list()[-1]
    y_true = K.cast(y_true > 0.5, dtype=y_true.dtype)
    y_pred = K.cast(y_pred > 0.5, dtype=y_pred.dtype)
    if label_length > 1:
        for num_label in range(label_length):
            y_true_f = K.flatten(y_true[...,num_label])
            y_pred_f = K.flatten(y_pred[...,num_label])
            intersection = K.sum(y_true_f * y_pred_f)
            notTrue = 1 - y_true_f
            union = K.sum(y_true_f + (notTrue * y_pred_f))
            loss += intersection / (union + K.epsilon())
        return loss / label_length
    else:
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        notTrue = 1 - y_true_f
        union = K.sum(y_true_f + (notTrue * y_pred_f))
        loss += intersection / (union + K.epsilon())
        return loss / label_length