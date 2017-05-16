# Vehicle Detection Project

The goals of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a car classifier.
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---

### 1. Feature Extraction

The first task is to define a feature extractor. This can be seen in the second code cell in the notebook.

I first grab an example `vehicle` and `non-vehicle`:

![](test_images/31.png)
![](test_images/extra1.png)

My feature extractor consists of three parts:

- **Histogram of Oriented Gradients (HOG)** features
- **Color features**
- **Spatial features**

To get the HOG feature vector, I use skimage's `hog()` function. After some experimentation, the parameters I ended up using are: 

```
orient = 9                    # HOG orientations
pix_per_cell = 8              # HOG pixels per cell
cell_per_block = 2            # HOG cells per block
hog_channel = 'ALL'           # 0, 1, 2, "GRAY" or "ALL"
```

To get the color features, I take a histogram of each color channel, with bin size 32, and concatenate these together into one vector.

The color feature histograms look like this on the above vehicle image:

![](output_images/1.png)

To get the spatial features, I use shrink the image down to 32x32 pixels and use `ravel()` to put the values into a vector.

The spatial features and gradient features look like this:

![](output_images/2.png)

And so the full feature vector extracted looks like the following. I run it through StandardScaler to normalize the features:
![](output_images/4.png)

The feature vector has 8,460 features.

### 2. Train the classifier

Next, I train the classifier over the 17,760 vehicle and non-vehicle images, making sure to normalize them first. This can be seen in the fourth code cell in the notebook. I acheive 98.56% accuracy using a 20% randomly selected validation set.

### 3. Sliding Window Search

Next up I define a function so that I can give it an area in an image and window size parameters and it will return co-ordinates of windows. I 

![](output_images/5.png)

### 4. Heatmapping

### 5. Video Implementation

Here's [my video result](./project_video.mp4).

### 6. Filter for false positives

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here are six frames and their corresponding heatmaps:

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![](output_images/6.png)

Here the resulting bounding boxes are drawn onto the last frame in the series:

![](output_images/7.png)

---

### Discussion

