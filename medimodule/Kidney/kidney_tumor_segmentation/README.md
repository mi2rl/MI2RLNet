# Kidney 
module 소개

### Results
메인 README.md에 결과가 있다면 표를 추가해줍니다.

## Kidney and Tumor Segmentation
submodule 소개

### Inference
```python
from medimodule.$MODULE import $SUBMODULE

module = $SUBMODULE()

# set the model with weight
module.init('/path/model/weights')

# get a result
result = module.predict('/path/of/image')

# get a kidney mask of the image
mask = module.predict('path/of/kidney_MASK.nii')
```
> 만약 샘플 결과가 있다면 이미지를 추가해줍니다.

### Weights
 - TODO



# KiTS19_ACE
KiTS 2019 challenge in MICCAI 2019  
Team name : ACE (Asan Coreline Ensemble)  
http://results.kits-challenge.org/miccai2019/manuscripts/sungchul7039_3.pdf
  
## Training
- **TO DO**

## Prediction
All checkpoints are located in `checkpoint/`. Checkpoints used in challenges will be updated.
- For searching ROI of kidney  
  `python evaluation.py --mode 1 --testset /path/testset`
- For predicting kidney and tumor  
  Select a mode using prediction. Before predicting kidney and tumor, **RUN** the mode 1 first.  
  2_1 : coreline's model
  2_2 : model with dice loss, normalization with tumor's mean and std and using **ONLY ONE** kidney in CT.  
  2_3 : model with dice loss, minmax scaling and using **ALL** kidney in CT.  
  2_4 : model with focaldice loss, minmax scaling and using **ALL** kidney in CT.  
  2_5 : model with dice loss, normalization with tumor's mean and std and using **ALL** kidney in CT.  
  `python evaluation.py --mode 2_3 --testset /path/testset`
