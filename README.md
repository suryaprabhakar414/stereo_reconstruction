# stereo-reconstruction

## Main
Command: ./stereo_reconstruction \[algorithm\] \[dataset\] \[number of test images\]

### Example

#### odometry
./stereo_reconstruction BM ODOMETRY 10

This command will run the stereo reconstruction full pipeline using Block Matching algorithms based on the 10 images in Kitti odemetry dataset sequences.

The result pose will be saved in txt.

#### stereo
./stereo_reconstruction BM STEREO 10

This command will calculate and save the disparity map using Block Matching algorithms based on the 10 images in Kitti stereo dataset.

The result disparity maps will be saved in png.

## Eval
Command: ./stereo_reconstruction_eval \[evaluation type\]

### Example

#### Pose Evaluation
./stereo_reconstruction_eval POSE

This command will go through all the supported algorithm to find their latest result pose and compare with the Kitti ground truth pose of odometry dataset.

The result of the evaluation could be found in the result folder.

#### Disparity Map Evaluation
./stereo_reconstruction_eval DISPARITY

This command will go through all the supported algorithm to find their latest result disparity map and compare with the Kitti ground truth disparity map of stereo dataset.

The result of the evaluation could be found in the result folder.
