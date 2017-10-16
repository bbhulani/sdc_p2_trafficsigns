#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
./CarND-Traffic-Sign-Classifier-Project/histogram.jpg ----> "Histogram of training data set"

Test images
./CarND-Traffic-Sign-Classifier-Project/road_work2.jpg
./CarND-Traffic-Sign-Classifier-Project/Stop_Sign.jpg
./CarND-Traffic-Sign-Classifier-Project/children_crossing.jpg
./CarND-Traffic-Sign-Classifier-Project/wild_animals.jpg
./CarND-Traffic-Sign-Classifier-Project/roundabout.png


## Rubric Points
---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

[TODO]
You're reading it! and here is a link to my https://github.com/bbhulani/sdc_p2_trafficsigns.git

###Data Set Summary & Exploration
I used the python methods to calculate summary statistics of the traffic
signs data set:

Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

####2. Include an exploratory visualization of the dataset.

Below is a histogram of the classes of the training data set ...
./CarND-Traffic-Sign-Classifier-Project/histogram.jpg

Preprocessing dataset
1. Grayscaled the image to reduce the image depth from 3 to 1
2. Normalized the image using zero mean to smoothen the transition of pixels
3. I didnt augment the data or increase the data set. The histogram clearly shows that the distribuition of training set is uneven and so making it more evenly distributed amongst the different classes would surely help. 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Model type: Modified LeNet architecture, increased depth of layers to extract more features. Maintained an apporximate ratio of 0.4 for fully connected layers

My final model consisted of the following 5 layers:

| Layer1         		|     Convolutional layer						| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
|:---------------------:|:---------------------------------------------:|
| Layer2        		|     Convolutional layer						| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 14x14x16 image   								| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x24 					|
|:---------------------:|:---------------------------------------------:|
| Layer3        		|     Fully connected layer						| 
|:---------------------:|:---------------------------------------------:|
| Input         		| 5x5x24 image   								| 
| Flatten 				| output = 600 									|
| Fully connected		| output = 240 									|
| RELU					|												|
| Dropout				|												|
|:---------------------:|:---------------------------------------------:|
| Layer4        		|     Fully connected layer						| 
|:---------------------:|:---------------------------------------------:|
| Input         		| 240    										| 
| Fully connected		| output = 100 									|
| RELU					|												|
| Dropout				|												|
|:---------------------:|:---------------------------------------------:|
| Layer5        		|     Fully connected layer						| 
|:---------------------:|:---------------------------------------------:|
| Input         		| 100    										| 
| Fully connected		| output = 43 									|
|						|												|

Output of Layer5 gives the logits which are then converted to softmax probabilities and run through cross entropy function with the one hot encodings. Next the loss is calulated with by averaging over the cross entropy function 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used the AdamOptimizer same as the LeNet architecture with a learning rate of 0.001
I kept the batch size at 128 and increased the number of epochs to 25 as the model was underfitting with a smaller epoch value
Next I fed a keep probability to 0.5 for the dropout function for Layer3 and layer 4 as the model was overfitting with 25 epochs

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I used the LeNet architecture to start out with. Then made the following modifications iteratively to improve the accuracy
1. Added normalization and greyscaling of dataset
2. Increased the number of epochs as the model was underfitting
3. Increased the depth of the convolutional layers to extract more features
4. Used regularization and added dropouts to fully connected layers to increase the accuracy
 
My final model results were:
* validation set accuracy of ~97% 
* test set accuracy of 94.7%


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

1. ./CarND-Traffic-Sign-Classifier-Project/road_work2.jpg
Should be easy to classify as the image is clear

2. ./CarND-Traffic-Sign-Classifier-Project/Stop_Sign.jpg
Should be easy to classify as the image is clear

3. ./CarND-Traffic-Sign-Classifier-Project/children_crossing.jpg
Might be harder as after resizing the image the pixels identifying children are not clear. A higher resolution image or an original 32z32 image would have been better

4. ./CarND-Traffic-Sign-Classifier-Project/wild_animals.jpg
Should be easy to classify as the image is clear

5. ./CarND-Traffic-Sign-Classifier-Project/roundabout.png
Might be harder to classify because after resizing the roundabout sign lost resolution and the pixels look different


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work   			| Road work 									|
| Stop Sign      		| Stop sign   									| 
| Children crossing		| Bicycles crossing								|
| Wild animals crossing	| Wild animals crossing 						|
| Roundabout mandatory	| Priority road      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60% vs the Test set accuracy of 95%. The downloaded images need to be resized correctly so that they dont loose resolution in order to improve the accuracy! 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image ... the model is relatively sure that this is a road work sign

The top five soft max probabilities were [25 30 22 29 20]

| Probability         	|     Prediction								| 
|:---------------------:|:---------------------------------------------:| 
| .636         			| Road work   									| 
| .091     				| Beware of ice/snow 							|
| .074					| Bumpy road									|
| .064					| Bicycles crossing					 			|
| .051					| Dangerous curve to the right      			|


For the second image ... the model is relatively sure that this is a stop sign
The top five soft max probabilities were  [14  5 38  3  1]
| Probability         	|     Prediction								| 
|:---------------------:|:---------------------------------------------:| 
| .470  				| Stop   										| 
| .158  				| Speed limit (80km/h) 							|
| .094					| Keep right									|
| .070					| Speed limit (60km/h)				 			|
| .019					| Speed limit (20km/h)      					|

For the third image ... the model is inaccurately sure that this is  bicyles crossing sign
The top five soft max probabilities were  [29 23 30 28 31]
| Probability         	|     Prediction								| 
|:---------------------:|:---------------------------------------------:| 
| .456 					| Bicycles crossing   							| 
| .315 					| Slippery road 								|
| .195					| Beware of ice/snow							|
| .031					| Children crossing				 				|
| .036					| Wild animals crossing      					|

For the 4th image ... The model is sure that this is wild animals crossing
The top five soft max probabilities were  [31 23 21 30 25]
| Probability         	|     Prediction								|
|:---------------------:|:---------------------------------------------:| 
| .99987 				| Wild animals crossing   						| 
| .00007 				| Slippery road 								|
| .00003				| Double curve									|
| .00003				| Beware of ice/snow				 			|
| .00001				| Road work      								|

For the 5th image ... the model is predicting with ONLY 30% chance its a priority road when the correct prediction would be mandatory aroundabout which only has a 10% probability 
The top five soft max probabilities were  [12 13  8 40 38]]
| Probability         	|     Prediction								| 
|:---------------------:|:---------------------------------------------:| 
| .295					| Priority road   								| 
| .164					| Yield 										|
| .162					| Speed limit (120km/h)							|
| .100					| Roundabout mandatory				 			|
| .007				    | Keep right      								|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


