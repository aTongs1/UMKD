# UMKD

## Datasets

Dataset   |        URL       
:--------------:|:------------------:|
眼底数据集       |   [APTOS_2019](https://www.kaggle.com/datasets/mariaherrerot/aptos2019)              
眼底数据集       |  [Eyepacs](https://zhuanlan.zhihu.com/p/683930522)        
前列腺癌数据集    |   [SICAPv2](https://zhuanlan.zhihu.com/p/686314573) 

## Results

### Teacher Performance
Teacher Model   |        Code     |    num_classes   
:--------------:|:------------------:|:--------------------:
ResNet50        |   resnet_linear_dr.py        |     眼底5 / 前列腺4             
ResNet50        |   resnet_linear_dr.py        |     眼底5 / 前列腺4          

### Student Performance (CUB200+StanfordDogs)
Target Model    |     Code       |      Methids 
:--------------:|:-----------:|:-------------------:
ResNet18        |   Resnet_trainer_student.py    |      common feature learning
ResNet18        |   Resnet_trainer_student_DKD.py    |      common feature learning + DKD 
ResNet18        |   Resnet_trainer_student_SDD.py    |      common feature learning + SDD  
ResNet18        |   Resnet_trainer_student_DKD_LP.py    |      common feature learning + DKD + LowPass  
ResNet18        |   Resnet_trainer_student_DKD_REDL.py    |      common feature learning + DKD + REDL  
ResNet18        |   Resnet_trainer_student_DKD_SA.py    |      common feature learning + DKD + ShallowAlign  
ResNet18        |   Resnet_trainer_student_DKD_SPP.py    |      common feature learning + DKD + SPP  
