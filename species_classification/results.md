# Report of Species Classification

In this markdown I will summarize the different approaches I tried to classify the animals per species. 

## Data Exploration
Here I will state some facts regarding the data that are relevant for this task. 

Number of images per split: 
* Train: 31,037 images
* Val: 5,414 images
* Test: 4,632 images
* Total: 41,083 images

Number of images per species per split:
* Train: 
    * class 0: 2,751 images -> 8.9 %
	* class 1: 10,048 images -> 32.4 %
	* class 2: 4,040 images -> 13.0 %
	* class 3: 13,496 images -> 43.5 %
	* class 4: 702 images -> 2.3 %
* Val: 
	* class 0: 1,732 images -> 32.0 %
	* class 1: 2,508 images -> 46.3 %
	* class 2: 214 images -> 4.0 %
	* class 3: 781 images -> 14.4 %
	* class 4: 179 images -> 3.3 %
* Test: 
	* class 0: 961 images -> 20.7 %
	* class 1: 1,552 images -> 33.5 %
	* class 2: 12 images -> 0.3 %
	* class 3: 1,943 images -> 41.9 %
	* class 4: 164 images -> 3.5 %
* Total: 
	* class 0: 5,444 images -> 13.3 %
	* class 1: 14,108 images -> 34.3 %
	* class 2: 4,266 images -> 10.4 %
	* class 3: 16,220 images -> 39.5 %
	* class 4: 1,045 images -> 2.5 %

Number of different flights per split:
* Train: 159 different flights
* Val: 24 different flights
* Test: 42 different flights
* Total: 225 flights

## Problems
The first problem is regarding the data distribution. There are only 12 samples of class 2 (Fallow Deer) in the test dataset. 


If flight A contains mostly roe deer at one altitude and flight B contains mostly red deer at another altitude, the network learns:

"this texture/background/scale = roe deer"

instead of

"this animal shape = roe deer"

Then train accuracy explodes while test accuracy stays terrible.

The fact that:

stratifying train gave 70% but original test still 40%

is actually evidence for this hypothesis.

The model performs well when train/val distributions match, but collapses when evaluated on unseen flights.


## Tried Approaches
In this section I want to quickly state every approach and model architecture I tried for the species classification. 

### Oversampling
As images of the classes 1 (Roe Deer) and 3 (Wild Boar) make up 75 % of the whole train dataset, I tried to mitigate this by oversampling the classes that are underrepresented. Unfortunately, it did not lead to validation and test accuracy scores over 40 %.

### Data Augmentation


### Make bigger crops
The surroundings of the animals play also a part in classifying the different species. Therefore, I decided to add more margin of the nature to the image crops. Unfortunately, it did not lead to validation and test accuracy scores over 40 %.

### Balanced Class Weights
While training the model I tried using balanced class weights that penalize misclassifications of smaller classes more than from the majority classes 1 and 3. Unfortunately, it did not lead to validation and test accuracy scores over 40 %.
TODO is that correct???

### Transfer Learning
For transfer learning I used the EfficientNetV2B0 model as it is one of the smaller models but reportedly still leads to good results. The problem is that the model was not trained on thermal image data, thus it also only led to accuracies for the validation and test dataset around 40 %. 

### RGB Images
As the model performances with only the thermal images were unfortunately bad, I added the RGB images to the output. The input arrays now are of the shape (128, 128, 4). 

This lead immeadiately to a rise in validation accuracy from around 40% to 65% and a balanced accuracy from 25% to 46%. Also the test accuracy increased to 49.9% and 35.7% balanced accuracy. 

## Conclusion
