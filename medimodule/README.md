> 모듈별 README template  
> 예제는 `medimodule/Liver/README.md` 참조하세요

# Title
module 소개

### Results

메인 README.md에 결과가 있다면 표를 추가해줍니다.

<br>

## Submodule title

submodule 소개



### Inference

```python
from medimodule.$MODULE import $SUBMODULE

# Set the model with weight
module = $SUBMODULE("path/of/weight")

# Get a result of the image
image, mask = model.predict("path/of/image", 
                            save_path="path/for/save")
```
> 만약 샘플 결과가 있다면 이미지를 추가해줍니다.



<br>

## Submodule title

### Inference

<br>

## Submodule title

### Inference
