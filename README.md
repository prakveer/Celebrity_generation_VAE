# Mask_RCNN on pedestrain and car dataset

This project implements a toned down version of mask rcnn on a toy dataset (pedestrian and car dataset). 
The complete mask rcnn can be divided into 5 parts. Each part builds upon the previous part
1. Region Proposal Network classifier: Predicts whether an object is present in that location.
2. Region Proposal Network regressor: Predicts the bounding box coordinates for that object.
3. ROI pooling: Currently, For the top 2 probable object locations (from rpn) , extracts and scales the  o/p of the base network according to the region of interest (from rpn regressor)
4. Faster RCNN: classifies the object into pedestrain or car
5. Mask RCNN: predicts the segementation mask for the object (whole network is trained)

For more details, refer the handout.

For results, refer the report.

### Prerequisites

Tensorflow verion > 1.4.0

OpenCV
## Running

* Kindly check the directories for the dataset in utils.py inside both RPN and RCNN folders.

* For example, to train the mask rcnn network , navigate to the RCNN directory.

```
python mask_rcnn.py
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* CIS 680 Advanced topics in perception - University of Pennsylvania
* CS20 stanford Tensorflow for Deep Learning Research 


