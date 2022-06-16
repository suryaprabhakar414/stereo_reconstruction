//
// Created by Ziyuan Qin on 13.01.22.
//

#include "registration.h"

void icp_registration(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in,
                      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out,
                      Eigen::Matrix4f &transformation_icp,
                      double max_correspondence,
                      int max_iterations,
                      double transformation_epsilon,
                      double euclidean_fitness_epsilon,
                      float outlier_threshold)
{
    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;

    // Set the input source and target
    icp.setInputSource(cloud_in);
    icp.setInputTarget(cloud_out);

    icp.setMaxCorrespondenceDistance(max_correspondence);
    icp.setMaximumIterations(max_iterations);
    icp.setTransformationEpsilon(transformation_epsilon);
    // icp.setEuclideanFitnessEpsilon(euclidean_fitness_epsilon);
    // icp.setRANSACOutlierRejectionThreshold(outlier_threshold);
    icp.setRANSACIterations(50);

    // perform the alignment
    pcl::PointCloud<pcl::PointXYZRGB> final;
    icp.align(final);

    // obtain the transformation that aligned cloud_source to cloud_source_registered
    transformation_icp = icp.getFinalTransformation ();

    if (icp.hasConverged()) {
        std::cout << "ICP has converged, score: " << icp.getFitnessScore() << std::endl;
        std::cout << "final transformation ICP: \n" << transformation_icp << std::endl;
    } else
        std::cout << "\nICP cannot converge" << std::endl;
}
