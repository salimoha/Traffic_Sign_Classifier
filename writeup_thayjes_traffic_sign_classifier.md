# **Traffic Sign Recognition** 




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

[train1]: ./writeup_images/TrainWA.jpeg "Visualization1"
[valid1]: ./writeup_images/ValidWA.jpeg "Visualization2"
[test1]: ./writeup_images/Test.jpeg "Visualization3"
[train2]: ./writeup_images/TrainA.jpeg "Visualization4"
[valid2]: ./writeup_images/ValidA.jpeg "Visualization5"
[image2]: ./writeup_images/Color_Traffic_Sign.png "Color"
[gray]: ./writeup_images/Gray_Traffic_Sign.PNG "Gray"
[augmented]: ./writeup_images/Augmented_Traffic_Sign.png "Augmented"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./writeup_images/Example_Traffic_Signs.png "Example Signs"
[fm]: ./writeup_images/FeatureMaps.png "Feature Maps"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/Thayjes/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3))
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. In the figures below, we can observe three bar plots. One for each dataset.
We can observe obviously a larger proportion of images in the training set when compared to the other two sets. Also another point to take note of is the unequal distribution of images in each class. This may make it difficult for the network to predict those classes with fewer examples. Later this can be dealt with using data augmentation for those classes with fewer samples.

![alt text][train1]  


![alt text][valid1]


![alt text][test1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

In my first step, I decided to convert the images to grayscale because this not only reduced the time it took to train. But also helped improve the accuracy.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2] 

![alt text][gray]

In my seconds step, I normalized the image data because this ensured all the pixel values lied between -0.5 and 0.5. This helps remove the differences between bright and dark images, thus reducing effect of contrast.

I decided to generate additional data because my initial model was overfitting the data.

To add more data to the the data set, I used the following techniques :
1. Translation: This involved translating each image by a small distance between (-1.5 and 1.5 pixels) in both the x and y directions.
2. Perspective Transform: Each image was warped using a perspective transform. This made the network more robust to the images from various angles.
3. Rotation: Finally, the image was rotated by a small angle between (-15 and 15 degrees). 
This data augmentation allows the network to generalize better to various traffic signs. Finally, the data was augmented based on the sample count of each label. So labels with lower sample count had more images augment to them. This ensured that the network was exposed to enough samples of each class.

Here is an example of an original image and an augmented image:

![alt text][augmented]

The difference between the original data set and the augmented data set is the following :
The distribution of samples in each class is much more evenly distributed. We can see compared to before, the augmented set has many more samples for those classes which were undersampled before.
![alt text][train2]

![alt text][valid2]

* The size of augmented training set is 99623
* The size of the augmented validation set is 24906

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16  |
| RELU					|												|
| Max Pooling		      | 2x2 stride, outputs 5x5x16       |
| Fully Connected 400x1			| outputs 120x1       									|
| RELU					|												|
| Dropout 1.0					|												|
| Fully Connected 120x1			| outputs 84x1       									|
| RELU					|												|
| Dropout 1.0					|												|
| Fully Connected 84x1			| outputs 43x1       									|
|	Softmax 43x1					|		outputs 43x1										|

 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an 
Optimizer: Adam Optimizer 
Batch Size: 128
Number of Epochs: 10
Learn Rate: 5*1e-4


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99%
* validation set accuracy of 98.2% 
* test set accuracy of 90.1%


* Which parameters were tuned? How were they adjusted and why?
The learn rate was tested over a few values, such as 1e-2, 1e-3 and 1e-4. The learning curves were too steep for the first two rates. And a tad bit slow for the final rate, but the curves looked a lot better. Finally I decided on a rate of 0.0005, as it was a good tradeoff.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
A dropout layer was included as this helps with improving an overfitted model to generalize.

If a well known architecture was chosen:
* LeNet architecture was chosen.
* A high training accuracy indicates that the weights have learned the training set well. A good validation accuracy indicates ability to generalize to new images. And finally a good test set accuracy implies how well the model will do in the real world.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image4] 

Here I will discuss some of the properties of the images that may make the difficult to be classified.
Road Work: The road work image is very blurry and has a lot of noise in it. The quality of the image is also poor. This may make the image difficult to classify.
No Vehicles: This sign has a very noisy background and there is no clear contrast unlike a few other images. This may be an obstacle for the network to overcome.
Finally
Speed Limit (60km/h): The color is a bit faded in the sign with respect to the other images.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit (30km/h)     		| Speed Limit (30km/h)   									| 
| Bumpy Road     			| Bumpy Road 										|
| Ahead Only					| Ahead Only											|
| No Vehicles	      		| No Vehicles				 				|
| Go Straight or Left			| Go Straight or Left      							|
| General Caution     | General Caution
| Road Work      | Road Work
| Speed Limit (60km/h) | Speed Limit (60km/h)


The model was able to correctly guess 8 of the 8 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 90%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .97       			| Speed Limit (30km/h)   									| 
| .98   				| Bumpy Road 										|
| .99					| Ahead Only										|
| .67	      			| No Vehicles					 				|
| .99				    | Go Straight or Left      							|
| .99 | General Caution
| .99 | Road Work
| .99 | Speed Limit (60km/h)


For the second image, the probability is 0.67 as the No Vehicles sign is not very distinct and has some similarities to other signs such as Stop Sign.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
Below is an image of the feature maps after the first layer of convolution.

![alt text][fm] 


We can see from the maps. In the first layer the network uses characteristics such as edges of circles and lines.


