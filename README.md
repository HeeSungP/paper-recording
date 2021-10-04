# paper-recording

## >< 건물 데이터 복원 파트

|Version|Accuracy|object IoU|back IoU|mIoU|
|:--------:|:------:|:------:|:------:|:------:|
|contour|93.076|92.014|65.002|78.508|
|base_e|94.967|93.790|78.189|85.989|
|base_c|-|-|-|-|
|base_raw|||||
|v1_2|94.210|93.122|71.761|82.442|
|v3_2|94.892|93.823|76.191|85.007|
|v4_1+2|95.140|94.085|77.495|85.790|
|v6_1+2|||||

## >< 창호 데이터 복원 파트

|Version|Accuracy|object IoU|back IoU|mIoU|
|:--------:|:------:|:------:|:------:|:------:|
|contour|77.739|76.058|24.130|50.094|
|base_e|87.498|84.180|62.612|73.396|
|base_raw|89.009|86.066|64.676|75.371|
|v4_1+2|86.854|83.576|60.599|72.088|
|v6_1+2|86.859|83.303|61.932|72.618|


## Version Description

### ■ base : contour segmentation
#### - contour : 모델 없이 max contour 비교
#### - base_e : dexi edge만 input으로
#### - base_c : max contour만 input으로

### ■ hs_v1_2 2021-09-23
#### - input : contour
#### - guide : dexi edge
#### - output : segmap

### ■ hs_v3_2 2021-09-22 
#### - input : contour + dexi edge
#### - guide : dexi edge
#### - output : segmap

### ■ hs_v4_1+2 2021-09-22~23
#### - input : contour + dexi edge
#### - guide : dexi edge
#### - output : segmap edge
#### >> cascade 구조를 위한 edge 복원 테스트 (ok)


#### - input : v1 contour + dexi edge
#### - guide : dexi edge
#### - output : segmap


### ■ hs_v61+2
#### - input : contour + dexi edge
#### - guide : dexi edge
#### - output : segmap edge
#### >> cascade 구조를 위한 edge 복원

#### - input : raw img + v1 result contour
#### - guide : v1 result contour
#### - output : segmap


# 제외한 실험
## - batch 4로 실험 통일하기 위해 제외

### ■ hs_v3 2021-09-22 15:07
#### - input : contour + dexi edge
#### - guide : dexi edge
#### - output : segmap
#### - setting : batch 1, epoch 50

### ■ hs_v1 2021-09-22 12:31
#### - input : contour
#### - guide : dexi edge
#### - output : segmap
#### |v1|95.358|94.330|78.740|86.535|


# 모델 관련 아이디어 메모

### 1. 2021-08-26 : unsupervised instance segmentation with contour & algorithm (based object, edge detection)
![image](https://user-images.githubusercontent.com/67678405/134759697-9f0fadcc-9b36-47c3-a998-ce6ddd5a3de7.png)

 - contribution 부족 및 단순 나열 느낌

### 2. 2021-09-15 : object edge inpainting based semantic segmentation
![image](https://user-images.githubusercontent.com/67678405/134759747-df875b0f-c1c3-497a-8932-09e4cdaf7483.png)

 - 모델 성능에 대한 불확실성 및 기존 연구 방향성을 활용하는 것이 좋겠다는 피드백

### 3. 2021-09-16 : bounding box prior + contour estimation + feature of object edge + u-net based instance segmentation
![image](https://user-images.githubusercontent.com/67678405/134759782-ab1fa576-87ff-4d5f-96cd-d8bd71b52360.png)

 - 객체별 edge의 특성을 반영하여 contour estimation의 결과를 보정하고 이를 바탕으로 각각 segmentation을 진행하는 instance segmentation 방식에 대한 연구


### 3번 방향성과 관련한 아이디어

 - contour 복원을 기반으로한 cascade network with u-net
 - edge map을 guide로 한 encoder-decoder network
 - 전체 건물과 창문은 각각 다른 학습 과정, 모델을 가짐


