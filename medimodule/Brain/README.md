# Brain
This **Brain** module consists of the following functions.
-  MRA BET(MR angiography brain extraction tool)
- (TODO) Brain Aneurysm Segmentation

### Results
| Modality | Part | Module | Results |
| --- | --- | --- | --- |
| MRA | Brain | MRA BET | - |
| MRA | Brain | Brain Aneurysm Segmentation | - |

&#160; 
## MRA BET
- `MRA BET` is the preprocessing tool to segment brain tissue in MRA images using the U-Net from the paper ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/abs/1505.04597)


### Inference

```python
from medimodule.Brain import MRA_BET

module = MRA_BET()

# set the model with weight
module.init(weight_path, gpu_number)

# get a BET mask of the image
mask = module.predict(data_path)
```
<img src=" " width="100%"></img>

### Weights
- TODO

### Reference
- [UNet] - [code](https://github.com/milesial/Pytorch-UNet)



&#160;  
## Brain Aneurysm Segmentation
