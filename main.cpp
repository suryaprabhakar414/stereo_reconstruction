//
// Created by Zhijie Yang on 14.12.21.
//
#include "disparity.h"
#include "registration.h"
#include <string>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include "disparity_eval.h"
#include "defs.h"
#include "triangulation_indirect.h"
#include "io_util.h"

template<typename T>
void visualize_pointcloud(typename pcl::PointCloud<T>::ConstPtr cloud) {
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<T>(cloud, "cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
    viewer->addCoordinateSystem(1.0);
    // event loop
    // Each call to spinOnce gives the viewer time to process events, allowing it to be interactive.
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::microseconds(100000));
    }
}

// Down sample cloud to down_sampled
void downsample(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr down_sampled, double voxel_size) {
    pcl::VoxelGrid<pcl::PointXYZRGB> grid;
    const float leaf = voxel_size;
    grid.setLeafSize(leaf, leaf, leaf);
    grid.setInputCloud(cloud);
    grid.filter(*down_sampled);
}

// Run the pipeline for stereo reconstruction
// Example: ./stereo_reconstruction BM ODOMETRY 10 0
// Example: ./stereo_reconstruction BM STEREO 10
int main(int argc, char** argv) {
    std::string algorithm;
    std::string dataset;
    int num_test_images;
    bool usingGroundTruth;

    if (argc == 5) {
        // Define which matching algorithm will be used: 0-SURF, 1-SIFT, 2-ORB, 3-BM, 4-SGBM, default is BM
        algorithm = std::string(argv[1]);

        // Define which subdataset will be used: 0-Kitti stereo 2015, 1-Kitti odometry, default is odometry
        // This is because when running evaluation, stereo 2015 has disparity map ground truth, and odometry has pose ground truth
        dataset = std::string(argv[2]);

        // Define how many pairs of image will be used in stereo 2015 or how many frames in sequences will be used in odometry, default is 10
        num_test_images = atoi(argv[3]);

        // Define if use the ground truth pose for reconstruction or not
        usingGroundTruth = atoi(argv[4]) > 0? true: false;
    } else if (argc == 4 && std::string(argv[2]) == "STEREO") {
        algorithm = std::string(argv[1]);
        dataset = std::string(argv[2]);
        num_test_images = atoi(argv[3]);
    }
    else {
       algorithm =  "BM";
       dataset = "ODOMETRY";
       num_test_images = 10;
       usingGroundTruth = false;
    }

    // Check if input is valid
    bool isInputValid = true;

    std::transform(algorithm.begin(), algorithm.end(),algorithm.begin(), ::toupper);
    std::transform(dataset.begin(), dataset.end(),dataset.begin(), ::toupper);
    int matching_algorithm = checkInputAlgorithmValid(algorithm);
    if (matching_algorithm < 0) {
        std::cout<< "Invalid algorithm input: " << dataset << ", please select from SURF, SIFT, ORB, BM, SGBM." << std::endl;
        isInputValid = false;
    }
    if (dataset != "STEREO" && dataset != "ODOMETRY") {
        std::cout<< "Invalid dataset input: " << dataset << ", please choose from STEREO, ODOMETRY." << std::endl;
        isInputValid = false;
    }
    if (dataset == "ODOMETRY" && matching_algorithm < 3) {
        std::cout<< "Invalid algorithm for stereo 2015: please choose from BM, SGBM." << std::endl;
        isInputValid = false;
    }
    if (num_test_images < 1 || num_test_images > 100) {
        std::cout<< "Invalid images num input: " << num_test_images << ", please choose from 1 - 100" << std::endl;
    }
    if (!isInputValid) {
        return 1;
    }


    /*
     * Pipeline start
     * Stage 0: Dataset preprocessing
     */
    // read intrinsic for stereo camera
    K << K_arr[0], K_arr[1], K_arr[2], K_arr[3], K_arr[4], K_arr[5], K_arr[6], K_arr[7], K_arr[8];
    // TODO: delete the debugging code
    std::cout << K << std::endl;

    // read input stereo images from dataset
    std::string left;
    std::string right;

    // Prepare output dir
    std::string outputDir = std::string("../result/").append(algorithm);

    // For stereo dataset, calculate disparity map only
    if (dataset == "STEREO") {
        left = "../images/stereo/left/";
        right = "../images/stereo/right/";

        for (int i = 0; i < num_test_images; i++) {
            // read input stereo images
            char prefix[256];
            sprintf(prefix, "%06d_10", i);

            std::string image_path = std::string(prefix) + ".png";
            std::string image_path_left = left + image_path;
            std::string image_path_right = right + image_path;

            cv::Mat img_left = imread(image_path_left, cv::IMREAD_GRAYSCALE);
            cv::Mat img_right = imread(image_path_right, cv::IMREAD_GRAYSCALE);

            cv::Mat disparity_norm;
            cv::Mat disparity = calcDisparity(img_left, img_right, (Algorithm) matching_algorithm);
            normalize(disparity, disparity_norm, 0, 80 * 256, CV_MINMAX, CV_16UC1);

            std::string dir = std::string("../result/").append(algorithm) + "/data/disp_0/";
            saveDisparityMap(dir, image_path, disparity_norm);
        }
    }
    else { // For odometry dataset
        left = "../images/odometry/left/";
        right = "../images/odometry/right/";

        // Relative and accumulated transforms
        std::vector<Eigen::Matrix4f> vec_tf_rel;
        std::vector<Eigen::Matrix4f> vec_tf_acc;
        std::vector<Eigen::Matrix4f> left_right_orb;
        std::vector<Eigen::Matrix4f> left_right_sift;
        std::vector<Eigen::Matrix4f> left_right_surf;
        vec_tf_rel.push_back(Eigen::Matrix4f().setIdentity());
        vec_tf_acc.push_back(Eigen::Matrix4f().setIdentity());
        pcl::PointCloud<pcl::PointXYZRGB> indirect_cloud;

        std::string init_image_path_left = left + "000000.png";
        std::string init_image_path_right = right + "000000.png";
        cv::Mat init_img_left_color = imread(init_image_path_left, cv::IMREAD_COLOR);

        cv::Mat init_img_left;
        cv::cvtColor(init_img_left_color, init_img_left, cv::COLOR_BGR2GRAY);
        cv::Mat init_img_right = imread(init_image_path_right, cv::IMREAD_GRAYSCALE);

        /*
         * Stage 1: Disparity Map computation (use and compare different matching algorithms)
         */
        cv::Mat init_disparity_norm;
        cv::Mat init_disparity = calcDisparity(init_img_left, init_img_right, (Algorithm) matching_algorithm);
        normalize(init_disparity, init_disparity_norm, 0, 80, CV_MINMAX, CV_16UC1);
        // normalize(init_disparity, init_disparity_norm, 0, 80 * 256, CV_MINMAX, CV_16S);

        //cv::imshow("disparity", init_disparity_norm);
        //cv::waitKey();

        /*
         * Stage 2 & 3: Triangulation and projection
        */
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr init_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        init_cloud = disparity2PointCloudRGB(init_disparity_norm, K, baseline, init_img_left_color).makeShared();

        std::vector<Eigen::Matrix4f> transforms;
        std::vector<Eigen::Matrix4f> poses;
        Eigen::Matrix4f pose;
        pose.setIdentity();
        poses.push_back(pose);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr last_cloud(init_cloud);
        // create a point cloud to be merged
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged_cloud(init_cloud);

        Eigen::Matrix4f tf_orb;
        Eigen::Matrix4f tf_sift;
        Eigen::Matrix4f tf_surf;
        indirect::triangulation_indirect(init_img_left, init_img_right, tf_orb, tf_sift, tf_surf);
        left_right_orb.emplace_back(tf_orb);
        left_right_sift.emplace_back(tf_sift);
        left_right_surf.emplace_back(tf_surf);

        // Load Kitti ground truth pose
        vector<Eigen::Matrix4f> gt_poses;
        if (usingGroundTruth) {
            int gt_poses_num = 0;
            std::string gt_file = "../images/odometry/00.txt";
            FILE *fp = fopen(gt_file.c_str(),"r");
            if (!fp) {
                std::cout << "Ground truth file not exist" << std::endl;
                return 1;
            }
            while (!feof(fp) && gt_poses_num < num_test_images - 1) {
                Eigen::Matrix4f P = Eigen::Matrix4f::Identity();
                if (fscanf(fp, "%f %f %f %f %f %f %f %f %f %f %f %f",
                           &P(0, 0), &P(0, 1), &P(0, 2), &P(0, 3),
                           &P(1, 0), &P(1, 1), &P(1, 2), &P(1, 3),
                           &P(2, 0), &P(2, 1), &P(2, 2), &P(2, 3))==12) {
                    std::cout << P << std::endl << std::endl;
                    gt_poses.push_back(P);
                    gt_poses_num++;
                }
            }
            fclose(fp);
        }


        /*
         * Stage 4: Motion Estimation & Point cloud registration
        */
        for (int i = 1; i < num_test_images; i++) {
            // read input stereo images
            char prefix[256];
            sprintf(prefix, "%06d", i);

            std::string image_path = std::string(prefix) + ".png";
            std::string image_path_left = left + image_path;
            std::string image_path_right = right + image_path;

            cv::Mat img_left_color = imread(image_path_left, cv::IMREAD_COLOR);

            cv::Mat img_left;
            cv::cvtColor(img_left_color, img_left, cv::COLOR_BGR2GRAY);
            cv::Mat img_right = imread(image_path_right, cv::IMREAD_GRAYSCALE);

            cv::Mat disparity_norm;
            cv::Mat disparity = calcDisparity(img_left, img_right, (Algorithm) matching_algorithm);
            normalize(disparity, disparity_norm, 0, 80, CV_MINMAX, CV_16UC1);
            // normalize(disparity, disparity_norm, 0, 80 * 256, CV_MINMAX, CV_16S);

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            cloud = disparity2PointCloudRGB(disparity_norm, K, baseline, img_left_color).makeShared();

            Eigen::Matrix4f transform;

            std::cout << "\nRunning test image " << i << std::endl;

            if (usingGroundTruth) {
                pose = gt_poses[i - 1];
            } else {
//                icp_registration(last_cloud, cloud, transform);
                transform.setIdentity();
                transforms.push_back(transform);
                pose = transform * pose;
                poses.push_back(pose);
                last_cloud = cloud;
            }

            // transform point cloud
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp(new pcl::PointCloud<pcl::PointXYZRGB>);
            // pcl::transformPointCloud(*cloud, *temp, pose.inverse());
            pcl::transformPointCloud(*cloud, *temp, pose);

            // down sample current point cloud
            // pcl::PointCloud<pcl::PointXYZRGB>::Ptr down_sampled(new pcl::PointCloud<pcl::PointXYZRGB>);
            // downsample(temp, down_sampled, 0.1f);

            Eigen::Matrix4f tf_orb;
            Eigen::Matrix4f tf_sift;
            Eigen::Matrix4f tf_surf;
            indirect::triangulation_indirect(img_left, img_right, tf_orb, tf_sift, tf_surf);
            left_right_orb.emplace_back(tf_orb);
            left_right_sift.emplace_back(tf_sift);
            left_right_surf.emplace_back(tf_surf);

            // merge point clouds
            *merged_cloud += *temp;
        }

        // save poses
        savePoses(outputDir, "poses", poses);
        savePoses(outputDir, "orb", left_right_orb);
        savePoses(outputDir, "sift", left_right_sift);
        savePoses(outputDir, "surf", left_right_surf);
        std::string pcd_file_name = std::to_string(num_test_images);
        if (usingGroundTruth) {
            pcd_file_name.append("_gt");
        }
        pcd_file_name.append(".pcd");
        pcl::io::savePCDFileASCII(pcd_file_name, *merged_cloud);
        // visualize merged point cloud
        // visualize_pointcloud<pcl::PointXYZRGB>(merged_cloud->makeShared());
    }

    return 0;
}
