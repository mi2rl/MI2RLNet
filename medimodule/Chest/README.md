# Chest
This **Chest** module consists of the following functions.
- L,R Mark Detection
- PA / Lateral / Others Classification
- Contrast / Non-Contrast Classification
  

### Results
| Modality  | Module | Results |
| ---  | --- | --- |
| X-ray  | Chest L,R Mark Detection | b0 mAP:99.28% |
| X-ray | PA / Lateral / Others Classification | Acc: 94% (extra validation) |
| CT | Contrast / Non-Contrast Classification | Acc: 96% (extra validation) |



## Chest L,R Mark Detection

 `Chest L,R Mark Detection` is to get the prediction box in chest X-ray.


### Inference

```python
from Chest import ChestLRDetection
detection = ChestLRDetection()
# set the model with weight
detection.init(args.weights)
# get a Prediction image
predict = detection.predict(args.img)
```

<img src="imgs/lr_sample.png" width="50%"></img>

### Weights

[weights link](https://drive.google.com/file/d/1WbZbDYDx7KxqhufiXh1u54q0DjZbYuew/view?usp=sharing)

### Reference

[EfficientDet] [Code](https://github.com/xuannianz/EfficientDet) - [Paper](https://arxiv.org/pdf/1911.09070.pdf)



## Enhanced / Non-Enhanced Classification  

This module can classify Enhanced / Non-Enhanced CT. 



### Inference

```python
from medimodule.Chest.module import EnhanceCTClassifier
from medimodule.utils import Checker

checker = Checker()
eCT_classifier = EnhanceCTClassifier()

# sanity check
checker.check_input_type('your input file', 'dcm')

# model init
eCT_classifier.init('your weights file')

# input image must have 2 channels.
# 1: Original
# 2: L550 / W737 (windowing)
predict = eCT_classifier.predict('your input file')
```

**Input image example**

<img src="imgs/enhance_sample.png" width="35%"></img>



### Weights

[weights link](https://drive.google.com/file/d/15S494ac3pUJSD6vEMJlSRi0Y42iM2OoG/view?usp=sharing)



## PA / Lateral / Others Classification  

This module can classify PA / Lateral / Others in X-ray.



### Inference

```python
from medimodule.Chest.module import ViewpointClassifier
from medimodule.utils import Checker

checker = Checker()
view_classifier = ViewpointClassifier()

# sanity check
checker.check_input_type('your input file', 'dcm')

# model init
view_classifier.init('your weights file')

# input
predict = view.predict('your input file')
```

**Input image example**

<img src="imgs/view_sample.png" width="70%"></img>



### Weights

[weights link](https://drive.google.com/file/d/1iCa-iwrek-efn_zSmFNrxdP5q_UOYuoK/view?usp=sharing)