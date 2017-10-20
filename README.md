## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[imageCarHog]: ./output_images/car_hog.png
[imageNotCarHog]: ./output_images/notcar_hog.png
[imageBoxes]: ./output_images/bboxes.png
[imageCarsFound]: ./output_images/cars_found.png
[imageHeatmap]: ./output_images/heatmap.png




## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This document is the writeup.

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The images are converted to HLS color space from RGB, and the Histogram of Oriented Gradient is calculated. The choice of the color space was made by comparing the 384 instances of the lienar SVM (Support Vector Machine) classifiers trained with different parameters. Here is the list of parameters chosen for training.

* Color space: HLS
* Orientations: 12
* pixels per cell: (8, 8)
* cells per block: (2, 2)

The features are extracted by calling the function `skimage.feature.hog()` (project.ipynb cell #2).

### Comparison of the HOG features of a car and a not-car images.

![alt text][imageCarHog]
![alt text][imageNotCarHog]


####2. Explain how you settled on your final choice of HOG parameters.

I have trained linear SVM classifiers for each combination of the following parameter values (project.ipynb cell #3).

* Color space: RGB, HSV, LUV, HLS, YUV, YCrCb.
* HOG channel: 0, 1, 2, 'All'.
* pixels per cell: 6, 8, 12, 14.
* Orientations: 6, 9, 12, 14.

The average scores of accuracy of the classifiers for each color space were as following:

| Color space|Avg score|
|:----------:|:-------:|
| LUV        |  0.995  |
| HLS        |  0.995  |
| YUV        |  0.994  |
| YCrCb      |  0.994  |
| HSV        |  0.993  |
| RGB        |  0.990  |

For the color space HLS, the average scores for other variables were as following:

|Orientation      |Avg score|   
|:----------:|:-------:|
|9       | 0.994 |
|6       | 0.995 |
|12      | 0.995 |
|14      | 0.996 |

|pixels per cell|Avg score|
|:----------:|:-------:|
|12   | 0.994 |
|6    | 0.994 |
|8    | 0.995 |
|14   | 0.996 |

|HOG channel    |Avg score|
|:----------:|:-------:|
|2       | 0.993 |
|1       | 0.994 |
|ALL     | 0.995 |
|0       | 0.997 |

I have chosen orientation=12, pixels per cell=8, and HOG channel='ALL'. Although they are not the best values at the tests, they look a good compromise between the performance and the memory efficiency.


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In additinon to the HOG, I also use the color histogram (`color_hist()`, project.ipynb cell #2) and the bin spatial features (`bin_spatial()`,  project.ipynb cell #2) for the training. The number of bins for the histograms are set to 32.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The window search is performed by `find_bboxes()` (project.ipynb cell #8). The function calculates the hog features in small windows, which will be reused for sliding window searches.

Since the cars further away look smaller, the size of the windows are changed depending on the y-corrdinate of the search. I define the locations of the top and bototm windows in the function `get_win_range()` (project.ipynb cell #9). The window sizes are also defined in pixels in the same function. From those ranges, a linear regression is performed by `get_winsize_ftn()` to find a function that defines the window size as a funtion of y-corrdinate of the bottom of the windows (project.ipynb cell #9). The number of rows is a free parameter, and I set it to 4. Then the list of the y-coordinates of the bottom of the boxes are calculated by `get_ylow_list()` (project.ipynb cell #9).

### The four rows of the search windows are shown in the following plot.

![alt text][imageBoxes]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Below are examples of the window search on the six test images. There are some false positives, which may be removed by  putting a threshold on a heat map (see the next section). The multiple boxes around the real cars can also be cleared up with the heat map. Furthermore, if the mothod is applied to the video images, multiple images can be used to reduce a lot of noise.

![alt text][imageCarsFound]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output_images/project_svm.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

For each frame in the video, a heat map is constructed from the bounding boxes that are identified as a car (`find_cars_heatmap()`, project.ipynb cell #10). The heat map is initialized as a zero matrix of the size of the camera image. Each bounding box increments the values of all the pixels in the box (`add_heat()`, project.ipynb cell #10). Only the pixels with a value above a threshold are selected to form the final boxes, using the function `scipy.ndimage.measurements.label()`.

### Here are the examples of the heat map applied to the six test images.

![alt text][imageHeatmap]

For processing the images in the video, the history of the heat maps is stored as an exponential moving average of each pixel in the heat map. (ProcImgAvg.\__call__(), project.ipynb cell #11)

$$S_t = \textit{w} h_t + (1 - \textit{w}) S_{t-1},$$

where $$S_t$$ is a moving average at time $$t$$, $$\textit{w}$$ is the smoothing constant, and $$h$$ is the value of a pixel's heat map value. I have set the weigth to 0.12 and the threshold on the averaged heat map to 0.2.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I have found that the success of the tracking greatly depend on the choice of the location and the size of the moving windows. This would not be a problem if the road is always flat, but when it is hilly, the search may fail. Another shortcoming of the method is the calculation speed. There are many windows in the search, and the windows overlap each other, so the calculation time increases quickly. Using other algorithms that do not depend on the moving window search may make the tracking more robust, and faster. One such methos is YOLO (You only look once), a object detection algorithm based on the convolutional neural networks. Another is SSD (Single Shot MultiBox Detector).

