# Multi-task learning in 3D cephalometry
------------------------------------------------------------------------------------------------------------------------------------------
This project is based on my internship experience in Arts et Métiers ParisTech & Materialise France which combines two different tasks in one deep learning model for landmark localization as well as semantic segmentation based on 3D u-net architecture. Database in this project is private due to commitment.A jupyter notebook as an example is available in this repo. You can also find the report and presentation for this project in the repo. 



# Workflow

1.landmark localization based on heatmap

Here a Gaussian heatmap is created for each landmark whcich corresponds to its location in the 3D volume, and we can use deep learning to reconstruct another heatmap to estimate the numeric coordinats afterwards 

![image](http://github.com/Wxy-24/3D_Cephalometry/raw/master/3D_cephalometry/img/workflow.png)  

2.model for multi-task learning

Here multi-task learning is implemented in a hard parameter sharing fashion before the final output layer of a 3d u-net

![image](http://github.com/Wxy-24/3D_Cephalometry/raw/master/3D_cephalometry/img/model.png)  

3.models for comparison

Besides original model, we adopt short cut in original convolutional blocks to test the performance of the other model(3d res u-net & 3d dense u net)  

![image](http://github.com/Wxy-24/3D_Cephalometry/raw/master/3D_cephalometry/img/comparison.png)  




# example of results

1.metrics for landmarks localization: 
Euclidean distance: 1.69 ± 0.74 

2.metrics for semantic segmentation:
Pixel accuracy: 99%
Dice coefficient: 0.90

 
