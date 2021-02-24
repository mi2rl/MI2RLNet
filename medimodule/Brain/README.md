# Brain
This **Brain** module consists of the following functions.
-  MRI BET (Brain Extraction Tool for T1-wegithed MRI, MR angiography)
-  Brain Blackblood segmentation

### Results
| Modality | Part | Module | Results |
| --- | --- | --- | --- |
| T1-weighted MRI | Brain | MRI_BET | 0.93 (DSC) |
| MRA | Brain | MRI_BET | 0.90 (DSC) |
| MRI | Brain | BlackbloodSegmentation | 0.83 (DSC)|


&#160; 
## MRI BET
- `MRI BET` is the preprocessing tool for skull stripping, or brain extraction in MR modalities(T1-weighted MRI, MRA) using the U-Net from the paper ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/abs/1505.04597)


### Inference

```python
from medimodule.Brain.module import MRI_BET

# Set the model with weight
model = MRI_BET("path/of/weight")

# Get a brain tissue mask of the image
# img_type : MRI modality(T1/MRA)
# save_path : set if you want to save the mask and the skull-stripped image
image, mask = model.predict("path/of/image", 
                            img_type="T1" or "MRA",
                            save_path="path/for/save")
```

#### Result of T1-weighted MRI BET
![sample](imgs/mri_bet.png)

#### Result of MRA BET
![sample](imgs/mra_bet.png)

### Reference
- [UNet] - [code](https://github.com/milesial/Pytorch-UNet)


&#160;  
## Blackblood Segmentation
- The objective of this `blackblood segmentation` is to get the black blood vessel in brain MRI.

### Inference
```python
from medimodule.Brain.module import BlackbloodSegmentation

# Set the model with weight
model = BlackbloodSegmentation("path/of/weight")

# Get a blackblood mask of the image
# save_path : set if you want to save the mask
image, mask = model.predict("path/of/image", 
                            save_path="path/for/save")
```

### Sample
![sample](imgs/blackblood.png)
