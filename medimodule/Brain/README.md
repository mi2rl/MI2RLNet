# Brain
This **Brain** module consists of the following functions.
-  MRI BET (Brain Extraction Tool for T1-wegithed MRI, MR angiography)
-  Brain Blackblood segmentation
- (TODO) Brain Aneurysm Segmentation

### Results
| Modality | Part | Module | Results |
| --- | --- | --- | --- |
| T1-weighted MRI | Brain | MRI BET | 93.35 (DSC%) |
| MRA | Brain | MRI BET | - |
| blackblood | Brain | Brain blackblood Segmentation | - |


&#160; 
## MRI BET
- `MRI BET` is the preprocessing tool for skull stripping, or brain extraction in MR modalities(T1-weighted MRI, MRA) using the U-Net from the paper ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/abs/1505.04597)


### Inference

```python
### MRI_BET Example 
from medimodule.Brain.module import MRI_BET
from medimodule.utils import Checker

check = Checker()
mri_bet = MRI_BET()

# Check if the input data type is nifti(.nii)
check.check_input_type('path/of/img.nii', 'nii')

# Allocate the gpu
check.set_gpu(gpu_number, framework='pytorch')

# Set the model with weight
# Choose an appropriate weight according to the data modality
mri_bet.init('path/of/weight.pth')

# Get a brain tissue mask of the input data
# img_type : MRI modality(T1/MRA)
# save_mask : set True if you want to save the binary bet mask
# save_stripping : set True if you want to save the skull-stripped image
mask = mri_bet.predict('path/input_img.nii', 
                            img_type='T1',
                            save_mask=True, 
                            save_stripping=True) 
```

#### result of T1-weighted MRI BET
<img src="imgs/mri_bet.png" width="100%"></img>

#### result of MRA BET
<img src="imgs/mra_bet.png" width="100%"></img>

### Weights
- TODO

### Reference
- [UNet] - [code](https://github.com/milesial/Pytorch-UNet)


&#160;  
## Blackblood Segmentation
### Inference
```python
from Brain.module import BlackbloodSegmentation
module = BlackbloodSegmentation()
# set the model with weight
module.init('weight.pth')

# get a liver mask of the image
mask = module.predict('/path/of/brain_seg.png')
```

### Sample
- To-do

### Model evaluation
UNet Scored 0.808 score 


&#160;  
## (Todo) Brain Aneurysm Segmentation
