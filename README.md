## Objective:

Understand the complex and interlinked processes responsible for the evolution of
the pan-Arctic permafrost polygonal tundra

## Research Questions:

What are the current extents of ice-wedge polygon landscape and ice-wedge
polygon successional stages across Arctic tundra ?

What are the climate drivers that cause the system to initialize the ice-wedge
degradation?

What are the sub-polygon to watershed scale responses and feedbacks that
determine the direction of change

## Methods:

Mask R-CNN algorithm :
* generates proposals (i.e., candidate object bounding boxes) after scanning the image
* predicts the class, bounding box, and binary mask for each region of interest (RoI)

Steps for mapping algorithm :

* Calibration of Mask R-CNN
* Tiling of satellite imagery
* Inferencing of input image patches
* Conversion of mask-to-polygon
* Elimination of duplicate polygons and composing unique polygons

## Code:

iwp_main.py: The main script that uses MPI and Multiprocessing to consume the images<br>
iwp_divideimg.py: It is for dividing large RS images <br>
iwp_inferenceimg.py: The inference process <br>
iwp_stitchshpfile.py: It is for composing shapefiles derived from small divided images
