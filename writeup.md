#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./test/11.jpeg =100x100 "Right-of-way at the next intersection"
[image2]: ./test/12.jpeg =100x100 "Priority road"
[image3]: ./test/13.jpeg =100x100 "Yield"
[image4]: ./test/18.jpeg =100x100 "General caution"
[image5]: ./test/23.jpeg =100x100 "Slippery road"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/mille-printemps/sdc-term1-project2-traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)


###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

`numpy` was enough to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799,
* The size of test set is 12630,
* The shape of a traffic sign image is (32, 32, 3),
* The number of unique classes/labels in the data set is 43.

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

The code shows a histogram of the training, validation and test data for each traffic sign. 


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

The preprocess includes

* Grayscale conversion
* Normalization to the image size in the training data

The grayscale conversion is done here because exact color information would not be an important factor to identify and classify traffic signs. This conversion also reduces the number of dimensions of an image from 3 to 1, which is more efficient from a complexity of calculation point of view. 

The normalization to the image size in the training data is included here to handle arbitrary size image later. The size normalization is necessary because the model is designed to process a fixed size (32x32x1) image. 


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The data had already been split into the sets of data for training, validation and testing. Those data sets were used for each purpose as they are. 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the fifth cell of the IPython notebook. 

My final model consisted of the following layers. This is based on LeNet model and the dropout technique is applied to avoid overfitting:

| Layer         		|     Description	        | 
|---------------------|---------------------------------------------| 
| Input         		| 32x32x1 grayscale image   |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6|
| RELU					|-|
| Max pooling	      	| 2x2 stride, outputs 14x14x6 |
| Dropout              |-|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16|
| RELU                 |-|
| Max pooling          | 2x2 stride, outputs 5x5x16 |
| Dropout              |-|
| Fully connected		| Input 400, Output 120 |
| RELU                 |-|
| Fully connected      | Input 120, Output 84 |
| RELU                 |-|
| Fully connected      | Input 84, Output # of classes|
| Softmax				|-|


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the sixth cell of the IPython notebook. 

The list of parameters used

* The learning rate is 0.001,
* The number of epochs is 90,
* The batch size is 512,
* The keep rate of the dropout is 0.7.

The learning rate is not changed in a training session. The dropout technique is added to avoid overfitting. The batch size may seem relatively large, but this is decided by trial-and-error to avoid overfitting. Since the dropout technique makes the behavior of a training session probabilistic, a relatively large number of epochs is assigned to meet the specification. 


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the sixth cell of the IPython notebook.

My final model results were:

* training set loss of 0.120
* validation set accuracy of 0.942
* test set accuracy of 0.934

The approach to find a solution was largely trial-and-error though LeNet model was chosen as a starting model. 

1. Tried out LeNet model to see how feasible it was for this problem. The model seemed to be working to some extent. Started investigating how to improve the model. 
2. Added a fragment of code to calculate a training loss and a validation loss for each epoch to see the status of the learning, i.e. overfitting or underfitting.
3. After a while, it was found that the training loss got relatively small, but the validation loss did not. So it was assumed that overfitting was occurring. 
4. Tried to simplify the model to avoid the overfitting. However, removing a layer caused underfitting. So decided to apply the dropout technique to the original model. 
5. The dropout technique worked for overfitting, but adjusting the keep rate was done by trial-and-error. After the trial-and-error, it was found that the learning process became unstable when the keep rate was smaller than 0.7 and the validation set accuracy did not converge even with relatively large number of epochs. 
6. It was also found that the batch size affected the validation set accuracy and increasing it to 512 made the learning process more stable.
7. Finally, it is found that 0.7 keep rate and 90 epochs produced 0.93 or more validation set accuracy constantly. 
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

The last image might be difficult to classify because it is squeezed into the square when its size is normalized. As a result, the ratio of edges of the triangle of the sign is skewed. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the eighth cell and ninth of the IPython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	       | 
|---------------------|---------------------------------------------| 
| Right-of-way at the next intersection | Right-of-way at the next intersection | 
| Priority road     	| Priority road	|
| Yield					| Yield			|
| General caution	    | General caution |
| Slippery Road			| Right-of-way at the next intersection |

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is an expected result considering the test accuracy is 0.923. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 10th cell of the IPython notebook.

For the first image, the model is sure that this is a sign of right-of-way at the next intersection (probability of 0.99). 

| Probability      	| Prediction |
|------------------|---------------------------------------|
| .99         		| Right-of-way at the next intersection |
| .0007    			| Beware of ice/snow |
| .0002				| Double curve |
| .00007	      	| Pedestrians |
| .00002			| Road narrows on the right|


For the second image, the model is sure this is a sign of priority road (probability of 0.93).

| Probability      	| Prediction |
|------------------|---------------------------------------|
| .93         		| Priority road |
| .02     			| End of all speed and passing limits |
| .02				| Speed limit (100km/h) |
| .01	      		| Speed limit (80km/h) |
| .008				| End of no passing by vehicles over 3.5 metric tons|

For the third image, the model is sure that this is a sign of yield (probability of 1).

| Probability      	| Prediction |
|------------------|---------------------------------------|
| 1.         		| Yield |
| .0    			| Speed limit (20km/h) |
| .0				| Speed limit (30km/h) |
| .0	      		| Speed limit (50km/h) |
| .0				| Speed limit (60km/h)|

For the forth image, the model is sure that this is a sign of general caution (probability of 0.99). The probabilities of the others is almost 0. 

| Probability      	| Prediction |
|------------------|---------------------------------------|
| .99         		| General caution |
| .0     			| Traffic signals |
| .0				| Pedestrians |
| .0	      		| Road narrows on the right |
| .0				| Speed limit (50km/h)|

For the fifth image, the model fails to identify this sign. This sign is for slippery road. However, the model says that it is "Right-of-way at the next intersection" (probability of 0.93). The probability is low even for the top one. This would indicate that the test image did not fit the training set or the learned model is still overfitting.

| Probability      	| Prediction |
|------------------|---------------------------------------|
| .93         		| Right-of-way at the next intersection |
| .03     			| Wild animals crossing |
| .01				| Road work |
| .009	      		| Traffic signals |
| .006				| Road narrows on the right	|
