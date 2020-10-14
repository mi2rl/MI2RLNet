from .utils import inject_tfkeras_modules, init_tfkeras_custom_objects
from .efficientnet import *

EfficientNetB0 = inject_tfkeras_modules(EfficientNetB0)
EfficientNetB1 = inject_tfkeras_modules(EfficientNetB1)
EfficientNetB2 = inject_tfkeras_modules(EfficientNetB2)
EfficientNetB3 = inject_tfkeras_modules(EfficientNetB3)
EfficientNetB4 = inject_tfkeras_modules(EfficientNetB4)
EfficientNetB5 = inject_tfkeras_modules(EfficientNetB5)
EfficientNetB6 = inject_tfkeras_modules(EfficientNetB6)
EfficientNetB7 = inject_tfkeras_modules(EfficientNetB7)

preprocess_input = inject_tfkeras_modules(preprocess_input)

init_tfkeras_custom_objects()
