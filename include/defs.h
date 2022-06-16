//
// Created by Zhijie Yang on 21.01.22.
//

#ifndef STEREO_RECONSTRUCTION_DEFS_H
#define STEREO_RECONSTRUCTION_DEFS_H
#include <opencv2/opencv.hpp>

static const float K_arr[9] = {7.188560000000e+02, 0.f, 6.071928000000e+02,
                               0.f, 7.188560000000e+02, 1.852157000000e+02,
                               0.f, 0.f, 1.f
};

static float baseline = 0.54;

static cv::Mat_<float> K(3, 3);

enum Algorithm{SURF, SIFT, ORB, BM, SGBM, NAIVE_BM};
#endif //STEREO_RECONSTRUCTION_DEFS_H
