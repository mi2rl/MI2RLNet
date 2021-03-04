# MI2RLNet

<p align="center">
    <img src="./imgs/overall_architecture.png" width="70%" height="70%">
</p>

This MI2RLNet is the hub of pretrained models in the medical domain. 

We hope MI2RLNet helps your downstream task.



* **Organizing Team** : MI2RL, Asan Medical Center(AMC), Seoul, Republic of Korea

* **Contributor** 

  * Commiter : Kyuri Kim, Jiyeon Seo, Jooyoung Park, Mingyu Kim, Kyungjin Cho, Daeun Kim, Yujin Nam.

  * Reviewer : Sungman Cho, Sungchul Kim.

  * Data Maintainer : Miso Jang, Namkug Kim.
    
    <br>

* **Docker images**

  * Dockerfille : tensorflow > 2.x, Pytorch 1.x

    <br>

## **Contents**

### **Data description**

| Modality  | Part        | Module                                 | Data Reference                                               |
| --------- | ----------- | -------------------------------------- | ------------------------------------------------------------ |
| X-ray     | Chest       | L/R Mark Detection                     | AMC                                                          |
| X-ray     | Chest       | PA / Lateral /Others Classification    | AMC                                                          |
| CT        | Chest       | Enhanced / Non-Enhanced Classification | AMC                                                          |
| CT        | Chest       | Lung Segmentation                      | AMC                                                          |
| CT        | Abdomen     | Kidnet & Tmuor Segmentation            | [KiTS 2019](https://kits19.grand-challenge.org/)              |
| CT        | Abdomen     | Liver Segmentation                     | AMC, [LiTS 2017](https://competitions.codalab.org/competitions/17094) |
| Endoscopy | Abdomen | Polyp Detection                        | [Kvsair-SEG](https://datasets.simula.no/kvasir-seg)                                                            |
| MR        | Brain       | Brain Extraction                       | AMC                                                          |
| MR        | Brain       | Blackblood Segmentation                | AMC                                                          |



### **Experiment results**

| Modality | Part        | Module                                 | Results   | Wiki                                                         | Weights                                                      | Framework |
| -------- | ----------- | -------------------------------------- | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------- |
| X-ray    | Chest       | L/R Mark Detection                     | 0.99 (mAP) | [link](https://github.com/mi2rl/MI2RLNet/tree/master/medimodule/Chest) | [link](https://drive.google.com/file/d/1WbZbDYDx7KxqhufiXh1u54q0DjZbYuew/view?usp=sharing) | TF 2.x  |
| X-ray    | Chest       | PA / Lateral / Others Classification   | 0.94 (Acc, external) | [link](https://github.com/mi2rl/MI2RLNet/tree/master/medimodule/Chest) | [link](https://drive.google.com/file/d/1iCa-iwrek-efn_zSmFNrxdP5q_UOYuoK/view?usp=sharing) | TF 2.x     |
| CT       | Chest       | Enhanced / Non-Enhanced Classification | 0.96 (Acc, external) | [link](https://github.com/mi2rl/MI2RLNet/tree/master/medimodule/Chest) | [link](https://drive.google.com/file/d/15S494ac3pUJSD6vEMJlSRi0Y42iM2OoG/view?usp=sharing) | TF 2.x     |
| CT       | Chest    | Lung Segmentation                      | 0.98 (DSC) | -                                                            | [link](https://drive.google.com/file/d/1zvmXbn8f16pWaFNS-SCuGw95cyl989xE/view?usp=sharing) | TF 2.x    |
| CT       | Abdomen | Kidney & Tumor Segmentation            | 0.83 (DSC) | [link](https://github.com/mi2rl/MI2RLNet/tree/master/medimodule/Kidney) | [link](https://drive.google.com/drive/folders/1lsMegnl5AeS90M7n1e-QYgYpr7vX-4yP?usp=sharing) | TF 2.x    |
| CT       | Abdomen | Liver Segmentation                     | 0.97 (DSC) | [link](https://github.com/mi2rl/MI2RLNet/tree/master/medimodule/Liver) | [link](https://drive.google.com/file/d/1oaURDlhh4K7S39XjxnaZShyLeUqvtbLC/view?usp=sharing) | TF 2.x    |
| Endoscopy | Abdomen | Polyp Detection             | 0.70 (DSC) | [link](https://github.com/mi2rl/MI2RLNet/tree/master/medimodule/Endoscopy) | [link](https://drive.google.com/file/d/1pwePgaYsDCAeNhHXvDgehP-4chQsAGtc/view?usp=sharing) | Pytorch   |
| MR       | Brain       | MRI/MRA BET (Brain Extration Tool)          | 0.95 (DSC) | [link](https://github.com/mi2rl/MI2RLNet/tree/master/medimodule/Brain) | [MRI](https://drive.google.com/file/d/1hditllnGF9PURJhqkN_wIUsM_cMv8_VM/view?usp=sharing) [MRA](https://drive.google.com/file/d/1JjzTaizHYouU0kQQMp4RYj3So9YLmoFT/view?usp=sharing) | Pytorch   |
| MR       | Brain       | Blackblood Segmentation               | 0.83 (DSC) | [link](https://github.com/mi2rl/MI2RLNet/tree/master/medimodule/Brain) | [link](https://drive.google.com/file/d/1LMPveqQybGh9EJD9nL1JwPinbUDJjz_y/view?usp=sharing) | TF 2.x    |

<br>

## **How can we use ?**
- The example code below applies to almost all modules. Some modules may require additional parameters.

### **Inference**

```python
from medimodule.Abdomen import LiverSegmentation

# Initialize the model.
# If pre-trained weight exists, enter it together when the model is assigned.
model = LiverSegmentation("/path/of/weight")

# Get a result.
# If you want to save the result, enter it with `save_path` kwargs.
image, mask = model.predict("/path/of/image", save_path="/path/for/save")
```



### **Transfer Learning**

```python
# Import any module you want to fine-tune.
from medimodule.Abdomen import LiverSegmentation

# Initialize the model with pre-trained weight.
model = LiverSegmentation("/path/of/weight")

# Construct your custom training code.
...
model.train()
...
```

<br>

## **Contributing**

If you'd like to contribute, or have any suggestions for these guidelines, you can contact us at namkugkim@gmail.com or open an issue on this GitHub repository.

All contributions welcome! All content in this repository is licensed under the Apache 2.0 license.
