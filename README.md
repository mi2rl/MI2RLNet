# Private Code House

* **Trello link** : https://trello.com/b/QFhmCEkV/mi2rl-codeteam
* **Contributor** 
  * Commiter : Kyuri Kim, Jiyeon Seo, Jooyoung Park, Mingyu Kim, Kyungjin Cho, Daeun Kim, Yoojin Nam.
  * Reviewer : Sungman Cho, Sungchul Kim.
  * Data Maintainer : Miso Jang.
    <br>
  
* **Docker images**
  * Dockerfille : tf > 2.x, Pytorch 1.x
  * Dockerfile_tf.1.15 : tf < 2.x, Pytorch 1.x
    <br>

## Contents

**Data description**

| Modality  | Part        | Module                                 | Device | Data Reference                                               |
| --------- | ----------- | -------------------------------------- | ------ | ------------------------------------------------------------ |
| X-ray     | Chest       | L/R Mark Detection                     | -      | Seoul Asan Medical Center Health Screening & Promotion Center |
| X-ray     | Chest       | PA / Lateral /Others Classification    | -      | Seoul Asan Medical Center Health Screening & Promotion Center |
| CT        | Chest       | Enhanced / Non-Enhanced Classification | -      | Seoul Asan Medical Center Health Screening & Promotion Center |
| CT        | Chest       | Lung Segmentation                      | -      | Seoul Asan Medical Center Health Screening & Promotion Center |
| CT        | Abdomen     | Kidnet & Tmuor Segmentation            | -      | KiTS 2019 (https://kits19.grand-challenge.org/)              |
| CT        | Abdomen     | Liver Segmentation                     | -      | LiTS 2017 (https://competitions.codalab.org/competitions/17094) |
| Endoscopy | Colonoscopy | Polyp Detection                        | -      | -                                                            |
| MR        | Brain       | Brain Extraction                       | -      | -                                                            |



**Experiment results**

| Modality | Part        | Module                                 | Results   | Wiki                                                         | Weights                                                      | Framework |
| -------- | ----------- | -------------------------------------- | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------- |
| X-ray    | Chest       | L/R Mark Detection                     | mAP : 99.28 | [link](https://github.com/mi2rl/private-code-house/tree/master/medimodule/Chest) | [link](https://drive.google.com/file/d/1WbZbDYDx7KxqhufiXh1u54q0DjZbYuew/view?usp=sharing) | TF <1.15  |
| X-ray    | Chest       | PA / Lateral / Others Classification   | Acc : 94.0 | [link](https://github.com/mi2rl/private-code-house/tree/master/medimodule/Chest) | [link](https://drive.google.com/file/d/1iCa-iwrek-efn_zSmFNrxdP5q_UOYuoK/view?usp=sharing) | Keras     |
| CT       | Chest       | Enhanced / Non-Enhanced Classification | Acc : 96.0 | [link](https://github.com/mi2rl/private-code-house/tree/master/medimodule/Chest) | [link](https://drive.google.com/file/d/15S494ac3pUJSD6vEMJlSRi0Y42iM2OoG/view?usp=sharing) | Keras     |
| CT       | Chest    | Lung Segmentation                      | -         | -                                                            | [link](https://drive.google.com/file/d/1UJ5FEZbBtn85b5hY04Ipb8eZvGkn-h8D/view?usp=sharing) | TF 2.x    |
| CT       | Abdomen | Kidney & Tumor Segmentation            | -         | [link](https://github.com/mi2rl/private-code-house/tree/master/medimodule/Kidney) | [link](https://drive.google.com/drive/folders/1lsMegnl5AeS90M7n1e-QYgYpr7vX-4yP?usp=sharing) | TF 2.x    |
| CT       | Abdomen | Liver Segmentation                     | DSC : 96.94 | [link](https://github.com/mi2rl/private-code-house/tree/master/medimodule/Liver) | [link](https://drive.google.com/file/d/1oaURDlhh4K7S39XjxnaZShyLeUqvtbLC/view?usp=sharing) | TF 2.x    |
| Endoscopy | Colonoscopy | Polyp Detection             | -         | [link](https://github.com/mi2rl/private-code-house/tree/master/medimodule/Endoscopy) | [link](https://drive.google.com/file/d/1pwePgaYsDCAeNhHXvDgehP-4chQsAGtc/view?usp=sharing) | Pytorch   |
| MR       | Brain       | MRA BET(Brain Extration Tool)          |           | [link](https://github.com/mi2rl/private-code-house/tree/master/medimodule/Brain) | - | Pytorch   |
| MR       | Brain       | Black-blood Segmentation               | -         | [link](https://github.com/mi2rl/private-code-house/tree/master/medimodule/Brain) | -                                                            | TF 2.x    |


