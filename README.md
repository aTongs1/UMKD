# UMKD

## Datasets

眼底数据集: [APTOS_2019](https://www.kaggle.com/datasets/mariaherrerot/aptos2019)  
眼底数据集: [Eyepacs](https://zhuanlan.zhihu.com/p/683930522)  
前列腺癌数据集: [SICAPv2](https://zhuanlan.zhihu.com/p/686314573)  

## Results

### Teacher Performance
Teacher Model   |        Code     |    num_classes   
:--------------:|:------------------:|:--------------------:
ResNet50        |   resnet_linear_dr.py        |     5/4             
ResNet50        |   resnet_linear_dr.py        |     5/4             

### Student Performance (CUB200+StanfordDogs)
Target Model    |    KD       |       CFL   
:--------------:|:-----------:|:-------------------:
ResNet34        |   0.7684    |      **0.7721**
ResNet50        |   0.7965    |      **0.7997** 
DenseNet121     |   0.7769    |      **0.7815**
