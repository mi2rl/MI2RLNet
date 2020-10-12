# Private Code House

* 이 저장소는 MI2RL 연구실의 내부 코드를 정리하기 위한 목적으로 만들어졌습니다.
* **Contributor** 
  * Commiter : 김규리, 서지연, 박주영, 김민규, 조경진, 김다은, 남유진
  * Reviewer : 조성만, 김성철
  * 데이터 공개 범위 논의 및 IRB :장미소

## Docker 

https://hub.docker.com/layers/mi2rl/mi2rl_image/latest/images/sha256-27b571a1237808b15796ca60e2076621d84266af159eebfb63951c825a58d0d5?context=repo



## Classification, Regression, Segmentation, Detection

| Modality | Part       | Module                                 | Results                                 | Paper | Weights |
| -------- | ---------- | -------------------------------------- | --------------------------------------- | ----- | ------- |
| X-ray    | ChestPA    | Age Regression                         | Acc: 70.4 (in 4years)                   | -     | link    |
| X-ray    | ChestPA    | Sex Clasisifcation                     | Acc: 99.6                               | -     |         |
| X-ray    | ChestPA/AP | Lung Segmentation                      | Acc:                                    | -     | link    |
| X-ray    | Chest      | Cardiomegaly Classification            | Acc: 91.4                               | -     | link    |
| X-ray    | ChestPA    | Osteoporosis Classification            | Acc: 83.0 (40세 이상)                   | -     | link    |
| X-ray    | Chest      | F/U Classification                     | -                                       | -     | -       |
| CT       | Chest      | Contrast / Non-Contrast Classification | Acc: 96.0                               | -     | link    |
| CT       | ?          | IPF Progression Prediction             | -                                       | -     | link    |
| CT       | Brain      | Anomaly Detection                      | -                                       | -     | link    |
| CT       | Lung       | Nodule Segmentation & Generation       | Dice: 0.813 <br />RMSE:  0.0057(nodule) | -     | link    |
| CT       | Lung       | Artery/Vein Seperation in Vessel Mask  | -                                       | -     | link    |
| CT       | Abdomen    | Kidney, Tumor Segmentation             | -                                       | -     | link    |
| CT       | Abdomen    | Liver Segmentation                     | -                                       | -     | link    |
| CT       | Abdomen    | Liver Vessel Segmentation              | -                                       | -     | link    |
| CT       | Abdomen    | Liver HCC Segmentation                 | -                                       | -     | link    |
| OCTA     | Retina     | Vessel Segmentation                    | -                                       | -     | -       |
| RGB      | Retina     | Optic disc Segmentation                | Acc: 100%                               | -     | link    |
| RGB      | Retina     | Vessel Segmentation                    | -                                       | -     | -       |
| RGB      | ENT        | ENT Instance Segmentation              | -                                       | -     | link    |
| RGB      | Face       | DO Not Touch your face                 | Acc: 94%                                | -     | link    |
| RGB      | -          | Hand Hygiene Monitoring                |                                         | link  | link    |
| MRI      | Brain      | Blackblood Segmentation                | Dice: 0.808                             | -     | link    |



## Preprocessing

| Modality | Part    | Module                    | Docker |
| -------- | ------- | ------------------------- | ------ |
| -        | -       | MedImageHandler           | -      |
| CT       | Chest   | Low-dose Median Filtering | link   |
| CT       | -       | Bone Extraction           | link   |
| CT       | Abdomen | Liver Registration        | link   |
| RGB      | Retina  | Circle Masking            | link   |



## Generation, Super Resolution, Image-to-Image

| Modality | Part         | Module                          | Docker |
| -------- | ------------ | ------------------------------- | ------ |
| X-ray    | Cephalometry | Cephalometry Generation         | link   |
| CT       | -            | Slice Thickness Superresolution | link   |
| RGB      | Retina       | Retina Generation               | link   |



## Complex System (Matlab)

| Modality | Part  | Module                                |
| -------- | ----- | ------------------------------------- |
| CT       | Chest | Fractal dimension: chest CT           |
| RGB      | -     | Fractal dimension: 2D grayscale image |



## Research

| Modality | Part  | Title         | Paper               |
| -------- | ----- | ------------- | ------------------- |
| X-ray    | Chest | Y label noise | DOI : 10.2196/18089 |

