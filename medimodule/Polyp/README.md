# Polyp

## Polyp Segmentation
### Running
```python
from medimodule.polyp import PolypSegmentation

module = PolypSegmentation()

# set the model with weight
module.init('weight.pth')

# get a liver mask of the image
mask = module.predict('/path/of/polyp.png')
```
### Sample
<img src="img/just_pic.png" width="100%"></img>

### Model evaluation
UNet Scored 0.7 Mean DSC score 

