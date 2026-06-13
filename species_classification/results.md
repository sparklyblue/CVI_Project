# Report of species classification

## Data Exploration

Number of images per split: 
* Train:  31037 images
* Val:  5414 images
* Test:  4632 images

Number of images per species per split:
* Train: 
    * class 0: 2751 images -> 0.089 %
	* class 1: 10048 images -> 0.324 %
	* class 2: 4040 images -> 0.130 %
	* class 3: 13496 images -> 0.435 %
	* class 4: 702 images -> 0.023 %
* Val: 
	* class 0: 1732 images -> 0.320 %
	* class 1: 2508 images -> 0.463 %
	* class 2: 214 images -> 0.040 %
	* class 3: 781 images -> 0.144 %
	* class 4: 179 images -> 0.033 %
* Test: 
	* class 0: 961 images -> 0.207 %
	* class 1: 1552 images -> 0.335 %
	* class 2: 12 images -> 0.003 %
	* class 3: 1943 images -> 0.419 %
	* class 4: 164 images -> 0.035 %

Number of different flights per split:
* Train: 159 different flights
* Val: 24 different flights
* Test: 42 different flights

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

TODO: 
* different image sizes (i dont think that will improve performance BUT maybe try smaller (<128, like 100) sizes)

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