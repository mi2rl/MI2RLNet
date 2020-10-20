# Chest
This **Chest** module consists of the following functions.
- L,R letter Detection
- (TODO) Receive each EfficientNet backbone as a parameter

### Results
| Modality  | Module | Results |
| ---  | --- | --- |
| X-ray  | Chest L,R Detection | b0 mAP:99.28% |


## Chest L,R Detection
The objective of this `Chest L,R letter Detection` submodule is to get the prediction box in chest X-ray.

### Inference

```python
from medimodule.Chest import ChestLRDetection
module = ChestLRDetection()

# set the model with weight
module.init(model_path)


# get a Prediction image
predict = module.predict(img_path)

# if you run test.py, just run.
python test.py --mode lr_detection --img sample.png --weights sample.h5 --save_path sample_dir/

```
![image](https://user-images.githubusercontent.com/46750574/95935344-f7d18800-0e0d-11eb-8ccf-3bcf075d82d6.png)

### Weights
- TODO

### Reference

[EfficientDet]
[code](https://github.com/xuannianz/EfficientDet)
[paper](https://arxiv.org/pdf/1911.09070.pdf)


