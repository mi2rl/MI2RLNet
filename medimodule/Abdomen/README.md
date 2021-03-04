# Abdomen
This **Abdomen** module consists of the following functions.
- Liver Segmentation
- Kidney & Tumor Segmentation
- Polyp Detection

## Results
| Modality | Part | Module | Results |
| --- | --- | --- | --- |
| CT | Abdomen | LiverSegmentation | 0.97 (DSC) |
| CT | Abdomen | KidneyTumorSegmentation | 0.83 (DSC) |
| Endoscopy | Abdomen | PolypDetection | 0.70 (DSC) |

<br>

---
## Liver Segmentation
- The objective of `Liver Segmentation` is to get the liver mask in abdominal CT.

### Inference
```python
from medimodule.Abdomen import LiverSegmentation

# Set the model with weight
model = LiverSegmentation("path/of/weight")

# Get a liver mask of the image
image, mask = model.predict("path/of/image", 
                            save_path="path/for/save")
```

### Sample
<img src="imgs/liver_segmentation.png" width="100%"></img>

<br>

---
## Kidney & Tumor Segmentation
- The objective of `Kidney & Tumor Segmentation` is to get the kidney and tumor mask in abdominal CT.

### Inference
```python
from medimodule.Abdomen import KidneyTumorSegmentation

# Set the model with weight
# weight is set to List about 6 models
# -> [1, 2_1, 2_2, 2_3, 2_4, 2_5]
model = KidneyTumorSegmentation("path/of/weight")

# Get a liver mask of the image
# save_path must be set with ONLY prefix. 
# When saving, suffix of each model, such as mode_1 or mode_2_1, will be attached.
image, mask = model.predict("path/of/image", 
                            save_path="path/for/save")
```

### Sample
<img src="imgs/kidney_tumor_segmentation.png" width="100%"></img>

<br>

---
## Polyp Segmentation
- The objective of `Polyp Segmentation` is to get the polyp mask in colonoscopy.

### Inference
```python
from medimodule.Abdomen import PolypDetection

# Set the model with weight
model = PolypDetection("path/of/weight")

# Get a liver mask of the image
image, mask = model.predict("path/of/image", 
                            save_path="path/for/save")

```

### Sample
<img src="imgs/polyp_detection.png" width="100%"></img>