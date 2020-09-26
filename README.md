# Private Code House

* 이 저장소는 MI2RL 연구실의 내부 코드를 정리하기 위한 목적으로 만들어졌습니다.
* **Contributor** 
  * 코드 정리 관련 : 조성만, 김성철, 김민규
  * IRB 및 공개 범위 등 데이터 관련 : 김민규, 장미소



## Contents (Module)

* Classification, Regression, Segmentation, Detection

| Modality | Part       | Module                                 | Results               | Paper | Weights | Docker |
| -------- | ---------- | -------------------------------------- | --------------------- | ----- | ------- | ------ |
| X-ray    | ChestPA    | Age Regression                         | Acc:                  | link  | link    | link   |
| X-ray    | ChestPA    | Sex Clasisifcation                     | Acc: 99.6             | link  |         |        |
| X-ray    | ChestPA/AP | Lung Segmentation                      | Acc:                  | link  | link    | link   |
| X-ray    | Chest      | Cardiomegaly Classification            | Acc: 91.4             | link  | link    | link   |
| X-ray    | ChestPA    | 골다공증                               | Acc: 83.0 (40세 이상) | link  | link    | link   |
| CT       | Chest      | Contrast / Non-Contrast Classification | Acc: 96.0             | link  | link    | link   |
| RGB      | Retina     | Optic disc Segmentation                | Acc: 100%             | link  | link    | link   |



## Contents (Preprocessing)

* Preprocessing

| Mdality | Part   | Module                    | Docker |
| ------- | ------ | ------------------------- | ------ |
| CT      | Chest  | Low-dose Median Filtering | link   |
| CT      | -      | Bone Extraction           | link   |
| RGB     | Retina | Circle Masking            | link   |