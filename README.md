# paper-recording

## >< 건물 데이터 복원 파트

|Version|Accuracy|object IoU|back IoU|mIoU|
|:--------:|:------:|:------:|:------:|:------:|
|base|93.076|92.014|65.002|78.508|
|base_e|94.967|93.790|78.189|85.989|
|base_c|-|-|-|-|
|v1|95.358|94.330|78.740|86.535|
|v3_2|94.892|93.823|76.191|85.007|
|v4_1+2|95.140|94.085|77.495|85.790|

### ■ base : contour segmentation
#### - base : 모델 없이 max contour 비교
#### - base_e : dexi edge만 input으로
#### - base_c : max contour만 input으로

### ■ hs_v1 2021-09-22 12:31
#### - input : contour
#### - guide : dexi edge
#### - output : segmap

### ■ hs_v3_2 2021-09-22 
#### - input : contour + dexi edge
#### - guide : dexi edge
#### - output : segmap
#### - setting : batch 4, epoch 50

### ■ hs_v4_1+2 2021-09-22~23
#### - input : contour + dexi edge
#### - guide : dexi edge
#### - output : segmap edge
#### >> cascade 구조를 위한 edge 복원 테스트 (ok)


#### - input : v1 contour + dexi edge
#### - guide : dexi edge
#### - output : segmap
#### - setting : batch 4, epoch 50




# - 제외한 실험

## - batch 4로 실험 통일

### ■ hs_v3 2021-09-22 15:07
#### - input : contour + dexi edge
#### - guide : dexi edge
#### - output : segmap
#### - setting : batch 1, epoch 50
