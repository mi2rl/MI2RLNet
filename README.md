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

| Modality | Part        | Module                                 | Results   | Wiki                                                         | Weights                                                      | Framework |
| -------- | ----------- | -------------------------------------- | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------- |
| X-ray    | Chest       | L/R Mark Detection                     | mAP : 99.28 | [link](https://github.com/mi2rl/private-code-house/tree/master/medimodule/Chest) | [link](https://drive.google.com/file/d/1WbZbDYDx7KxqhufiXh1u54q0DjZbYuew/view?usp=sharing) | TF <1.15  |
| X-ray    | Chest       | PA / Lateral / Others Classification   | -         | [link](https://github.com/mi2rl/private-code-house/tree/master/medimodule/Chest) | [link](https://drive.google.com/file/d/1iCa-iwrek-efn_zSmFNrxdP5q_UOYuoK/view?usp=sharing) | Keras     |
| CT       | Chest       | Enhanced / Non-Enhanced Classification | Acc : 96.0 | [link](https://github.com/mi2rl/private-code-house/tree/master/medimodule/Chest) | [link](https://drive.google.com/file/d/15S494ac3pUJSD6vEMJlSRi0Y42iM2OoG/view?usp=sharing) | Keras     |
| CT       | Lung        | Lung Segmentation                      | -         | -                                                            | [link](https://drive.google.com/file/d/1UJ5FEZbBtn85b5hY04Ipb8eZvGkn-h8D/view?usp=sharing) | TF 2.x    |
| CT       | Kidney      | Kidney & Tumor Segmentation            | -         | [link](https://github.com/mi2rl/private-code-house/tree/master/medimodule/Kidney) | -                                                            | TF 2.x    |
| CT       | Liver       | Liver Segmentation                     | DSC : 96.94 | [link](https://github.com/mi2rl/private-code-house/tree/master/medimodule/Liver) | [link](https://drive.google.com/file/d/1oaURDlhh4K7S39XjxnaZShyLeUqvtbLC/view?usp=sharing) | TF 2.x    |
| RGB      | Colonoscopy | Poly Segmentation                      | -         | [link](https://github.com/mi2rl/private-code-house/tree/master/medimodule/Endoscopy) | [link](https://drive.google.com/file/d/1pwePgaYsDCAeNhHXvDgehP-4chQsAGtc/view?usp=sharing) | Pytorch   |
| MR       | Brain       | MRA BET(Brain Extration Tool)          |           | [link](https://github.com/mi2rl/private-code-house/tree/master/medimodule/Brain) | -                                                            | Pytorch   |
| MR       | Brain       | Black-blood Segmentation               | -         | [link](https://github.com/mi2rl/private-code-house/tree/master/medimodule/Brain) | -                                                            | TF 2.x    |


