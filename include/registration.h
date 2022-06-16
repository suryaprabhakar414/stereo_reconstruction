//
// Created by Ziyuan Qin on 13.01.22.
//

#ifndef POINT_CLOUD_REGISTRATION_H
#define POINT_CLOUD_REGISTRATION_H

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

// void icp_registration(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in,
//                       pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_out,
//                     //   pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud_ptr,
//                     //   Eigen::Matrix4f cloud_source_registered,
//                     //   int sky_direction,
//                       Eigen::Matrix4f &transformation_icp,
//                       double max_correspondence=0.1, // set the max correspondence distance to 10 cm (correspondences with higher distances will be ignored)
//                       int max_iterations=50, // set the maximum number of iterations (criterion 1)
//                       double transformation_epsilon=1e-6, // set the transformation epsilon (criterion 2)
//                       double euclidean_fitness_epsilon=1, // set the euclidean distance difference epsilon (criterion 3)
//                       float outlier_threshold=0.05);

void icp_registration(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in,
                      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out,
                      Eigen::Matrix4f &transformation_icp,
                      double max_correspondence=1, // set the max correspondence distance with unit meter (correspondences with higher distances will be ignored)
                      int max_iterations=70, // set the maximum number of iterations (criterion 1)
                      double transformation_epsilon=1e-6, // set the transformation epsilon (criterion 2)
                      double euclidean_fitness_epsilon=0.005, // set the euclidean distance difference epsilon (criterion 3)
                      float outlier_threshold=0.1);

#endif //POINT_CLOUD_REGISTRATION_H
