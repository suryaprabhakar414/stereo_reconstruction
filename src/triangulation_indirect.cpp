//
// Created by Zhijie Yang on 21.01.22.
//

#include "triangulation_indirect.h"
#include <opencv2/xfeatures2d/nonfree.hpp>
namespace indirect {
    cv::Point2f pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
        return cv::Point2f
                (
                        (p.x - K.at<float>(0, 2)) / K.at<float>(0, 0),
                        (p.y - K.at<float>(1, 2)) / K.at<float>(1, 1)
                );
    }

    void find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2,
                              std::vector<cv::KeyPoint> &keypoints_1,
                              std::vector<cv::KeyPoint> &keypoints_2,
                              std::vector<cv::DMatch> &matches,
                              Detector detector_type) {
        // Init
        cv::Mat descriptors_1, descriptors_2;
        cv::Ptr<cv::FeatureDetector> detector;
        cv::Ptr<cv::DescriptorExtractor> descriptor;
        std::string matcher_name;
        switch (detector_type) {
            case ORB:
                /// Not suitable for Flann based matching!!
                std::cout << "Running with ORB" << std::endl;
                detector = cv::ORB::create();
                matcher_name = "BruteForce-Hamming";
                break;
            case SIFT:
                /// Not suitable for Hamming based matching!!
                std::cout << "Running with SIFT" << std::endl;
                detector = cv::SIFT::create();
                matcher_name = "BruteForce";
                break;
            case SURF:
                /// Not suitable for Hamming based matching!!
                std::cout << "Running with SURF" << std::endl;
                detector = cv::xfeatures2d::SURF::create();
                matcher_name = "FlannBased";
                break;
        }
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(matcher_name);
        // 1. Detect corners
        // 2. Calculate descriptors
        detector->detectAndCompute(img_1, cv::noArray(), keypoints_1, descriptors_1);
        detector->detectAndCompute(img_2, cv::noArray(), keypoints_2, descriptors_2);
        if (matcher_name == "BruteForce-Hamming" || matcher_name == "BruteForce-Hamming(2)") {
            // 3. Match the descriptors with Hamming distance
            std::vector<cv::DMatch> match;
            // BFMatcher matcher ( NORM_HAMMING );
            matcher->match(descriptors_1, descriptors_2, match);

            // 4. Compute the min_ and max_dist
            double min_dist = 10000, max_dist = 0;

            // When the distance between two descriptors are larger than 2x min_dist, take is as an error.
            // An empirical value of 30 when min_dist is too small
            for (int i = 0; i < descriptors_1.rows; i++) {
                double dist = match[i].distance;
                if (dist < min_dist) min_dist = dist;
                if (dist > max_dist) max_dist = dist;
            }

            printf("-- Max dist : %f \n", max_dist);
            printf("-- Min dist : %f \n", min_dist);

            for (int i = 0; i < descriptors_1.rows; i++) {
                if (match[i].distance <= cv::max(2 * min_dist, 30.0)) {
                    matches.push_back(match[i]);
                }
            }
//        } else if (matcher_name == "BruteForce" || matcher_name == "BruteForce-L2") {
        } else {
            std::cout << "KNN matches on Flann" << std::endl;
            std::vector< std::vector<cv::DMatch> > knn_matches;
            matcher->knnMatch( descriptors_1, descriptors_2, knn_matches, 2 );
            //-- Filter matches using the Lowe's ratio test
            const float ratio_thresh = 0.75f;
            for (size_t i = 0; i < knn_matches.size(); i++)
            {
                if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
                {
                    matches.push_back(knn_matches[i][0]);
                }
            }
            std::cout << std::endl;

        }
    }

    void pose_estimation_2d2d(
            const std::vector<cv::KeyPoint> &keypoints_1,
            const std::vector<cv::KeyPoint> &keypoints_2,
            const std::vector<cv::DMatch> &matches,
            cv::Mat &R, cv::Mat &t) {

        std::vector<cv::Point2f> points1;
        std::vector<cv::Point2f> points2;

        for (auto match: matches) {
            points1.push_back(keypoints_1[match.queryIdx].pt);
            points2.push_back(keypoints_2[match.trainIdx].pt);
        }
        K << K_arr[0], K_arr[1], K_arr[2], K_arr[3], K_arr[4], K_arr[5], K_arr[6], K_arr[7], K_arr[8];
        // Compute essential matrix
        cv::Point2d principal_point(K.at<float>(1, 2), K.at<float>(0, 2));
        std::cout << "Principal point: " << principal_point << std::endl;
        int focal_length = K.at<float>(0, 0);
        cv::Mat essential_matrix;
        essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);

        // Recover rotation and translation from E
        recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    }

    void triangulation(
            const std::vector<cv::KeyPoint> &keypoint_1,
            const std::vector<cv::KeyPoint> &keypoint_2,
            const std::vector<cv::DMatch> &matches,
            const cv::Mat &R, const cv::Mat &t,
            std::vector<cv::Point3d> &points) {
        cv::Mat T1 = (cv::Mat_<float>(3, 4) <<
                                            1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0);
        cv::Mat T2 = (cv::Mat_<float>(3, 4) <<
                                            R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0,
                                                                                                                     0),
                R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
                R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
        );

        std::vector<cv::Point2f> pts_1, pts_2;
        for (cv::DMatch m: matches) {
            pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
            pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
        }

        cv::Mat pts_4d;
        cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

        // Change to homogeneous coordinates
        for (int i = 0; i < pts_4d.cols; i++) {
            cv::Mat x = pts_4d.col(i);
            x /= x.at<float>(3, 0); // normalization
            cv::Point3d p(
                    x.at<float>(0, 0),
                    x.at<float>(1, 0),
                    x.at<float>(2, 0)
            );
            points.push_back(p);
        }
    }

    inline cv::Scalar get_color(float depth) {
        float up_th = 50, low_th = 10, th_range = up_th - low_th;
        if (depth > up_th) depth = up_th;
        if (depth < low_th) depth = low_th;
        return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
    }

    pcl::PointCloud<pcl::PointXYZI> triangulation_indirect(const cv::Mat& img_1, const cv::Mat& img_2, Eigen::Matrix4f &tf_orb, Eigen::Matrix4f &tf_sift,
                                                           Eigen::Matrix4f &tf_surf) {
        std::vector<cv::KeyPoint> orb_keypoints_1, orb_keypoints_2;
        std::vector<cv::DMatch> orb_matches;
        find_feature_matches(img_1, img_2, orb_keypoints_1, orb_keypoints_2, orb_matches, Detector::ORB);
        std::cout << orb_matches.size() << " pairs of matched points are found" << std::endl;

        // Estimate the transform between two images
        cv::Mat R_orb, t_orb;
        pose_estimation_2d2d(orb_keypoints_1, orb_keypoints_2, orb_matches, R_orb, t_orb);
        Eigen::Matrix3f R_orb_eigen;
        Eigen::Vector3f t_orb_eigen;
        cv2eigen(R_orb, R_orb_eigen);
        cv2eigen(t_orb, t_orb_eigen);
        tf_orb.block(0, 0, 3, 3) = R_orb_eigen;
        tf_orb.block(0, 3, 3, 1) = t_orb_eigen;
        tf_orb.block(3, 0, 1, 4) = Eigen::Matrix<float, 1, 4>(0.0, 0.0, 0.0, 1.0);


        std::vector<cv::KeyPoint> sift_keypoints_1, sift_keypoints_2;
        std::vector<cv::DMatch> sift_matches;
        find_feature_matches(img_1, img_2, sift_keypoints_1, sift_keypoints_2, sift_matches, Detector::SIFT);
        std::cout << sift_matches.size() << " pairs of matched points are found" << std::endl;

        // Estimate the transform between two images
        cv::Mat R_sift, t_sift;
        pose_estimation_2d2d(sift_keypoints_1, sift_keypoints_2, sift_matches, R_sift, t_sift);
        Eigen::Matrix3f R_sift_eigen;
        Eigen::Vector3f t_sift_eigen;
        cv2eigen(R_sift, R_sift_eigen);
        cv2eigen(t_sift, t_sift_eigen);
        tf_sift.block(0, 0, 3, 3) = R_sift_eigen;
        tf_sift.block(0, 3, 3, 1) = t_sift_eigen;
        tf_sift.block(3, 0, 1, 4) = Eigen::Matrix<float, 1, 4>(0.0, 0.0, 0.0, 1.0);

        std::vector<cv::KeyPoint> surf_keypoints_1, surf_keypoints_2;
        std::vector<cv::DMatch> surf_matches;
        find_feature_matches(img_1, img_2, surf_keypoints_1, surf_keypoints_2, surf_matches, Detector::SURF);
        std::cout << surf_matches.size() << " pairs of matched points are found" << std::endl;

        // Estimate the transform between two images
        cv::Mat R_surf, t_surf;
        pose_estimation_2d2d(surf_keypoints_1, surf_keypoints_2, surf_matches, R_surf, t_surf);
        Eigen::Matrix3f R_surf_eigen;
        Eigen::Vector3f t_surf_eigen;
        cv2eigen(R_surf, R_surf_eigen);
        cv2eigen(t_surf, t_surf_eigen);
        tf_surf.block(0, 0, 3, 3) = R_surf_eigen;
        tf_surf.block(0, 3, 3, 1) = t_surf_eigen;
        tf_surf.block(3, 0, 1, 4) = Eigen::Matrix<float, 1, 4>(0.0, 0.0, 0.0, 1.0);

        // triangulation
        std::vector<cv::Point3d> points;
        triangulation(orb_keypoints_1, orb_keypoints_2, orb_matches, R_orb, t_orb, points);

        // Verify the reprojection correspondence between triangulated points and feature points
        cv::Mat img1_plot = img_1.clone();
        cv::Mat img2_plot = img_2.clone();
        for (int i = 0; i < orb_matches.size(); i++) {
            float depth1 = points[i].z;
//        std::cout << "depth: " << depth1 << std::endl;
//        Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
//        cv::circle(img1_plot, keypoints_1[matches[i].queryIdx].pt, 2, get_color(depth1), 2);
            cv::circle(img1_plot, orb_keypoints_1[orb_matches[i].queryIdx].pt, 2, cv::Scalar(255, 0, 255), 2);

            cv::Mat pt2_trans = R_orb * (cv::Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t_orb;
            float depth2 = pt2_trans.at<double>(2, 0);
//        cv::circle(img2_plot, keypoints_2[matches[i].trainIdx].pt, 2, get_color(depth2), 2);
            cv::circle(img2_plot, orb_keypoints_2[orb_matches[i].trainIdx].pt, 2, cv::Scalar(255, 0, 255), 2);
        }
//    cv::imshow("img 1", img1_plot);
//    cv::imshow("img 2", img2_plot);
//    cv::waitKey();
        pcl::PointCloud<pcl::PointXYZI> cloud;
        for (auto &p: points) {
            pcl::PointXYZI cloud_p;
            cloud_p.x = p.x;
            cloud_p.y = p.y;
            cloud_p.z = p.z;
            cloud_p.intensity = 0;
            cloud.push_back(cloud_p);
        }
        return cloud;
    }
}



