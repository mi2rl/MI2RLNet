# Liver
This **Liver** module consists of the following functions.
- Liver Segmentation

### Results
| Modality | Part | Module | Results |
| --- | --- | --- | --- |
| CT | Abdomen | LiverSegmentation | 0.97 (DSC) |


## Liver Segmentation
- The objective of this `Liver Segmentation` submodule is to get the liver mask in abdominal CT.

### Inference

```python
from medimodule.Liver import LiverSegmentation

# Set the model with weight
model = LiverSegmentation("path/of/weight")

# Get a liver mask of the image
image, mask = model.predict("path/of/image", 
                            save_path="path/for/save")
```
<img src="imgs/liver_segmentation.png" width="100%"></img>
