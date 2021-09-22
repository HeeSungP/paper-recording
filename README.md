# paper-recording

## >< 건물 데이터 복원 파트

|Version|Accuracy|object IoU|back IoU|mIoU|
|:--------:|:------:|:------:|:------:|:------:|
|base|93.076|92.014|65.002|78.508|
|v1|95.358|94.330|78.740|86.535|
|v3|95.509|94.495|79.594|87.044|

### ■ base : contour segmentation

### ■ hs_v1 2021-09-22 12:31
#### - input : contour
#### - guide : dexi edge
#### - output : segmap

### ■ hs_v3 2021-09-22 15:07
#### - input : contour + dexi edge
#### - guide : dexi edge
#### - output : segmap
#### - setting : batch 1, epoch 50

### ■ hs_v3_2 2021-09-22 
#### - input : contour + dexi edge
#### - guide : dexi edge
#### - output : segmap
#### - setting : batch 4, epoch 50

### ■ hs_v4 2021-09-22 (log 기록하는거 깜빡...)
#### - input : contour + dexi edge
#### - guide : dexi edge
#### - output : segmap edge
#### >> cascade 구조를 위한 edge 복원 테스트 (ok)


