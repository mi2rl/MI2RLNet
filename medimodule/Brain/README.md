# Brain
This **Brain** module consists of the following functions.
-  MRA BET(MR angiography brain extraction tool)
-  Brain Blackblood segmentation

- (TODO) Brain Aneurysm Segmentation

### Results
| Modality | Part | Module | Results |
| --- | --- | --- | --- |
| MRA | Brain | MRA BET | - |
| blackblood | Brain | Brain blackblood Segmentation | - |
| MRA | Brain | Brain Aneurysm Segmentation | - |

&#160; 
## MRA BET
- `MRA BET` is the preprocessing tool to segment brain tissue in MRA images using the U-Net from the paper ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/abs/1505.04597)


### Inference

```python
### MRA_BET Example 
from medimodule.Brain.module import MRA_BET
from medimodule.utils import Checker

check = Checker()
mra_bet = MRA_BET()

# check if the input data type is nifti(.nii)
check.check_input_type('path/of/img.nii', 'nii')
# allocate the gpu
check.set_gpu(gpu_number, framework='pytorch')

# set the model with weight
mra_bet.init('path/of/weight.pth')
# get a brain tissue mask of the input data(put the saving path if you want to save the output mask)
mask = mra_bet.predict('path/of/img.nii', 'save_path')
```

<img src="imgs/mra_bet.png" width="100%"></img>

### Weights
- TODO

### Reference
- [UNet] - [code](https://github.com/milesial/Pytorch-UNet)


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

## Brain Aneurysm Segmentation
