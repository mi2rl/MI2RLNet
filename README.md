# Private Code House

* 이 저장소는 MI2RL 연구실의 내부 코드를 정리하기 위한 목적으로 만들어졌습니다.
* **Contributor** 
  * 조성만, 김성철, 김민규, 조경진, 장미소





## Classification, Regression, Segmentation, Detection

| Modality | Part       | Module                                      | Results                                 | Paper | Weights | Docker |
| -------- | ---------- | ------------------------------------------- | --------------------------------------- | ----- | ------- | ------ |
| X-ray    | ChestPA    | Age Regression                              | Acc: 70.4 (in 4years)                   | -     | link    | link   |
| X-ray    | ChestPA    | Sex Classification                          | Acc: 99.6                               | -     |         |        |
| X-ray    | ChestPA/AP | Lung Segmentation                           | Acc:                                    | -     | link    | link   |
| X-ray    | Chest      | Cardiomegaly Classification                 | Acc: 91.4                               | -     | link    | link   |
| X-ray    | ChestPA    | Osteoporosis Classification                 | Acc: 83.0 (40세 이상)                   | -     | link    | link   |
| X-ray    | Chest      | F/U Classification                          | -                                       | -     | -       | link   |
| CT       | Chest      | Contrast / Non-Contrast Classification      | Acc: 96.0                               | -     | link    | link   |
| CT       | ?          | IPF Progression Prediction                  | -                                       | -     | link    | link   |
| CT       | Brain      | Anomaly Detection                           | -                                       | -     | link    | link   |
| CT       | Lung       | Nodule Segmentation & Generation            | Dice: 0.813 <br />RMSE:  0.0057(nodule) | -     | link    | link   |
| CT       | Lung       | Artery/Vein Seperation in Vessel Mask       | -                                       | -     | link    | link   |
| CT       | Abdomen    | Kidney, Tumor Segmentation                  | -                                       | -     | link    | link   |
| CT       | Abdomen    | Liver Segmentation                          | -                                       | -     | link    | link   |
| CT       | Abdomen    | Hepatic Vessel Segmentation                 | -                                       | -     | link    | link   |
| CT       | Abdomen    | HCC (Hepatocellular carcinoma) Segmentation | -                                       | -     | link    | link   |
| OCTA     | Retina     | Vessel Segmentation                         | -                                       | -     | -       | -      |
| RGB      | Retina     | Optic disc Segmentation                     | Acc: 100%                               | -     | link    | link   |
| RGB      | Retina     | Vessel Segmentation                         | -                                       | -     | -       | link   |
| RGB      | ENT        | ENT Instance Segmentation                   | -                                       | -     | link    | link   |
| RGB      | Face       | DO Not Touch your face                      | Acc: 94%                                | -     | link    | link   |
| RGB      | -          | Hand Hygiene Monitoring                     |                                         | link  | link    | link   |
| MRI      | Brain      | Blackblood Segmentation                     | Dice: 0.808                             | -     | link    | link   |



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

