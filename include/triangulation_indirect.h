//
// Created by Zhijie Yang on 21.01.22.
//

#ifndef STEREO_RECONSTRUCTION_TRIANGULATION_INDIRECT_H
#define STEREO_RECONSTRUCTION_TRIANGULATION_INDIRECT_H
#include <opencv2/opencv.hpp>
#include "defs.h"
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Core>

namespace indirect {
    enum Detector {
        SURF, SIFT, ORB
    };

    static const char* Matcher[] = {"BruteForce",
                       "BruteForce-L1",
                       "BruteForce-Hamming",
                       "BruteForce-Hamming(2)",
                       "FlannBased"
    };

    pcl::PointCloud<pcl::PointXYZI> triangulation_indirect(const cv::Mat& img_1, const cv::Mat& img_2, Eigen::Matrix4f &tf_orb, Eigen::Matrix4f &tf_sift,
                                                           Eigen::Matrix4f &tf_surf);
    /**
    @param matcher_name
    -   `BruteForce` (it uses L2 )
    -   `BruteForce-L1`
    -   `BruteForce-Hamming`
    -   `BruteForce-Hamming(2)`
    -   `FlannBased`
    **/
    void find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2,
                              std::vector<cv::KeyPoint> &keypoints_1,
                              std::vector<cv::KeyPoint> &keypoints_2,
                              std::vector<cv::DMatch> &matches,
                              Detector detector_type);
}
#endif //STEREO_RECONSTRUCTION_TRIANGULATION_INDIRECT_H
