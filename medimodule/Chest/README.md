# Chest_LR_Detection
This **Chest** module consists of the following functions.
- L,R letter Detection
- (TODO) Receive each EfficientNet backbone as a parameter

### Results
| Modality  | Module | Results |
| ---  | --- | --- |
| X-ray  | Chest L,R Detection | b0(base) mAP:99.28% |


### Inference

```python
from medimodule.Chest import ChestLRDetection
module = ChestLRDetection()

# set the model with weight
module.init(model_path,gpu_num,threshold)


# get a Prediction image
predict = module.predict(img_path,threshold)

# if you run test.py, just run.
python test.py --mode lr_detection --img sample.png --weights sample.h5 --save_path sample_dir/

```

### Results
![image](https://user-images.githubusercontent.com/46750574/95935344-f7d18800-0e0d-11eb-8ccf-3bcf075d82d6.png)

### Weights
- TODO

  
### Next 

- Each EfficientNet backbone as a parameter
- tensorflow-2.3 version apply

### Reference

[EfficientDet]-[code](https://github.com/xuannianz/EfficientDet)[paper](https://arxiv.org/pdf/1911.09070.pdf)