# Vehicle Detection Project

The goal of this project is to detect cars in a video feed and draw bounding boxes around them.

We do this with the following steps:

* Define a feature extractor. We use a Histogram of Oriented Gradients (HOG) feature extractor, a color feature extractor, and a spatial feature extractor. 
* Train a classifier on our labeled training set of images of cars and non-cars.
* Search windows in images for cars using the trained classifier.
* Run on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Draw a bounding box around vehicles detected.


### 1. Feature Extraction

The first task is to define a feature extractor. This can be seen in the second code cell in the notebook.

I first grab example `vehicle` and `non-vehicle` images:

![](test_images/31.png)
![](test_images/extra1.png)

My feature extractor consists of three parts:

- **Histogram of Oriented Gradients (HOG) features**
- **Color features**
- **Spatial features**

To get the HOG feature vector, I use skimage's `hog()` function. The parameters I ended up using are: 

```
orient = 9                    # HOG orientations
pix_per_cell = 8              # HOG pixels per cell
cell_per_block = 2            # HOG cells per block
hog_channel = 'ALL'           # 0, 1, 2, "GRAY" or "ALL"
```

I tried other parameters, such as 0, and ALL channels, but found they did not perform as well as GRAY.

After initially trying RGB color space, I switched it to YCrCb, and found that the classifier performed much better at detecting the cars and eliminating false positives in the images and videos, so I went with that.
The gradient feature vector run on the above vehicle image ends up looking like this:

![](output_images/9.png)

To get the color features, I take a histogram of each color channel, with bin size 32, and concatenate these together into one vector. The color feature histograms look like this:

![](output_images/1.png)

To get the spatial features, I use shrink the image down to 32x32 pixels and use `ravel()` to put the values into a vector. The spatial features look like this:

![](output_images/2.png)

And so the full feature vector extracted looks like the following. I run it through `StandardScaler` to normalize the features:

![](output_images/4.png)

### 2. Train the classifier

Next, I train the classifier over the 17,760 vehicle and non-vehicle images, making sure to normalize them first. This can be seen in the fourth code cell in the notebook. I use a LinearSVC, and it takes about 40 seconds to train it over the training set. I get 99% accuracy using a 20% randomly selected validation set.

After initially using simple just the `prediction()` function to choose the hot windows, I ended up using `decision_function()` function as well and chose the windows with only a large distance away from the decision line. This resulted in much better elimination of false positives in the test videos.

### 3. Sliding Window Search

Next up I define a function so that I can give it an area in an image and window size parameters and it will return coordinates of windows. I pass it the area from 45% down the image, which is just above the horizon, because cars can't fly yet, all the way down to 90% down the image, where the hood of the car is covering the road. I use 64, 96, 128, 256, 350, 512 pixel sqaure windows, each with 75% overlap.

For each frame of the video, I run the classifer over each of the windows defined by the function, and see if a car is detected in that window.

![](output_images/6.png)

### 4. Heatmapping

The classifier is not perfect, and often predicts false positives. So in order to work around this, I define a global `heat` variable that remains persistent frame by frame over the video. When the classifer finds a car in a window, I add the value of 1 to all the pixels in that window area in the `heat` image. I then threshold the heat image into a new image used to calculate the bounding boxes, so that only places in the image where the classifer detected a car in the same area many times in a frame or over multiple frames are shown. Finally I fade the heatmap away by 10% each frame so that when cars move around, they are no longer detected in that area.  I also found and fixed an issue with heat thresholding before persisting the heatmap from frame to frame, which greatly improved frame to frame detections.

### 5. Bounding boxes

After getting the heatmap, I use SciPy's `label()` to find where the bounding boxes are for the detections, by running it over the `heat` image. I draw these boxes over the original image like thus:

![](output_images/7.png)

### 6. Video Implementation

After running the pipeline one frame at a time over the video, I end up with bounding boxes drawn on each frame where cars were detected. 

Here's [my video result](./project_output.mp4).

### 7. Discussion

The end video quickly finds the white car and draws a bounding box over it. The position of the bounding box is not always exactly around where the car is though.

There are a number of false positives that remain detected over multiple frames of the videos, such as up in the trees and the road dividers.

This could be improved by improving the classifier and tuning the feature extractor.
