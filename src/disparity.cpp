//
// Created by Zhijie Yang on 14.12.21.
//

#include "disparity.h"
#include "pdist2.h"
#include "triangulation_indirect.h"
#include <opencv2/core/eigen.hpp>
#include <Eigen/Core>

// help functions declaration
bool checkThreshold(std::vector<std::pair<float, int>> distPairs, float threshold);

// matching algorithms declaration
void apply_stereo_BM(const cv::Mat &left_img, const cv::Mat &right_img, cv::Mat &disp_img, int max_disparity, int block_size);
void apply_stereo_SGBM(const cv::Mat &left_img, const cv::Mat &right_img, cv::Mat &disp_img, int min_disparity, int max_disparity, int block_size,
                       int P1, int P2, int disp12MaxDiff, int preFilterCap, int uniquenessRatio,
                       int speckleWindowSize, int speckleRange, int mode);
void apply_naive_block_matching(const cv::Mat &left_img, const cv::Mat &right_img, cv::Mat &disp_img, int block_size, int min_disparity, int max_disparity,
                             float d);
/*
 * Returns the disparity map of two input stereo images
 *
 */
cv::Mat calcDisparity(const cv::Mat& left_img, const cv::Mat& right_img, Algorithm matching_algorithm) {
    cv::Mat disp_img = cv::Mat::zeros(left_img.size(), left_img.type());

    // param list for matching algorithms
    int min_disparity = 0;
    int max_disparity = 80;
    int BM_block_size = 15;
    int SGBM_block_size = 13;
    // is used in naive BM algorithm to check if the founded correspondence is valid
    float validDisparityThreshold = 1.5;
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    std::vector<cv::DMatch> matches;
    cv::Mat img_matches;
//    std::string matcher_name = "BruteForce-Hamming";

    switch(matching_algorithm) {
        case SURF:
        case SIFT:
        case ORB:
            std::cout << "Running with indirect methods" << std::endl;
            find_feature_matches(left_img, right_img, keypoints_1, keypoints_2, matches, (indirect::Detector) matching_algorithm);
            std::cout << matches.size() << "pairs of matched points are found" << std::endl;
            std::cout << keypoints_1.size() << "key points in left image" << std::endl;
            std::cout << keypoints_2.size() << "key points in right image" << std::endl;
            drawMatches(left_img, keypoints_1, right_img, keypoints_2, matches, img_matches, cv::Scalar::all(-1),
                        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            //-- Show detected matches
            cv::imshow("Good Matches", img_matches);
            cv::waitKey();
            break;
        case NAIVE_BM:
            std::cout << "Running with naive BM" << std::endl;
            apply_naive_block_matching(left_img, right_img, disp_img, BM_block_size,
                                            min_disparity, max_disparity,validDisparityThreshold);
            break;
        case SGBM:
            std::cout << "Running with SGBM" << std::endl;
            apply_stereo_SGBM(left_img, right_img, disp_img, min_disparity, max_disparity, SGBM_block_size,
                              8 * left_img.channels() * SGBM_block_size * SGBM_block_size, 32 * left_img.channels() * SGBM_block_size * SGBM_block_size,
                              1, 50, 0, 50, 1, false);
            break;
        case BM:
        default:
            std::cout << "Running default: BM" << std::endl;
            apply_stereo_BM(left_img, right_img, disp_img, max_disparity, BM_block_size);
            break;
    }

    return disp_img;
}

/*
 * TODO: determine 3D point cloud for disparity image in camera frame
 * conventions:
 * - points: pcl point cloud
 * - intensities: grey values of the corresponding pixels
 * - only pixels with valid disparity should be included into the point cloud
 */
pcl::PointCloud<pcl::PointXYZRGB> disparity2PointCloudRGB(const cv::Mat& disp_img, cv::Mat K, float baseline, const cv::Mat& left_img) {
    unsigned int n_valid;
    auto width = left_img.size().width;
    auto height = left_img.size().height;
    auto K_inv = K.inv();

    pcl::PointCloud<pcl::PointXYZRGB> points;

    auto fx = K.at<float>(0, 0);
    auto fy = K.at<float>(1, 1);
    auto cx = K.at<float>(0, 2);
    auto cy = K.at<float>(1, 2);

    cv::Mat disp_img_32f;
    disp_img.convertTo(disp_img_32f, CV_32F, 1.0);
    // disp_img.convertTo(disp_img_32f, CV_32F, 1.0 / 16.0);

    for (int v = 0; v < left_img.rows; v++) {
        uchar* rgb_ptr = const_cast<uchar *>(left_img.ptr<uchar>(v));
        for (int u = 0; u < left_img.cols; u++) {
            if (disp_img_32f.at<float>(v, u) <= 1.0 || disp_img_32f.at<float>(v, u) >= 96.0) continue;
            if (disp_img_32f.at<float>(v, u) <= 1.0 || disp_img_32f.at<float>(v, u) >= 360.0) continue;
            float x = (u - cx) / fx;
            float y = (v - cy) / fy;
            float depth = fx * baseline / (disp_img_32f.at<float>(v, u));
            if (depth > 15.0f)
                continue;

            pcl::PointXYZRGB p;
            p.x = x * depth;
            p.y = y * depth;
            p.z = depth;

            uchar pb = rgb_ptr[3 * u];
            uchar pg = rgb_ptr[3 * u + 1];
            uchar pr = rgb_ptr[3 * u + 2];

            uint32_t rgb = (static_cast<uint32_t>(pr) << 16 |
                            static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));
            p.rgb = *reinterpret_cast<float*>(&rgb);

            points.push_back(p);
        }
    }

    return points;
}


/*
pcl::PointCloud<pcl::PointXYZI> disparity2PointCloud(const cv::Mat& disp_img, cv::Mat K, float baseline, const cv::Mat& left_img) {
    unsigned int n_valid;
    auto width = left_img.size().width;
    auto height = left_img.size().height;
    auto K_inv = K.inv();

    pcl::PointCloud<pcl::PointXYZI> points;

    auto fx = K.at<float>(0, 0);
    auto fy = K.at<float>(1, 1);
    auto cx = K.at<float>(0, 2);
    auto cy = K.at<float>(1, 2);

    cv::Mat disp_img_32f;
    disp_img.convertTo(disp_img_32f, CV_32F, 1.0 / 16.0f);

    for (int v = 0; v < left_img.rows; v++)
        for (int u = 0; u < left_img.cols; u++) {
            if (disp_img_32f.at<float>(v, u) <= 1.0 || disp_img_32f.at<float>(v, u) >= 96.0) continue;

            float x = (u - cx) / fx;
            float y = (v - cy) / fy;
            float depth = fx * baseline / (disp_img_32f.at<float>(v, u));
            pcl::PointXYZI p;
            p.x = x * depth;
            p.y = y * depth;
            p.z = depth;
            p.intensity = left_img.at<float>(v, u) / 255;
            points.push_back(p);

        }

    return points;
}
*/

/*
 * Matching algorithms implementation
 */
void apply_stereo_BM(const cv::Mat &left_img, const cv::Mat &right_img, cv::Mat &disp_img, int max_disparity, int block_size) {
    auto stereoBM = cv::StereoBM::create(max_disparity, block_size);
    stereoBM->compute(left_img, right_img, disp_img);
}

void apply_stereo_SGBM(const cv::Mat &left_img, const cv::Mat &right_img, cv::Mat &disp_img, int min_disparity, int max_disparity, int block_size,
                       int P1, int P2, int disp12MaxDiff, int preFilterCap, int uniquenessRatio,
                       int speckleWindowSize, int speckleRange, int mode) {
    auto stereoSGBM = cv::StereoSGBM::create(min_disparity,max_disparity,block_size,
                                        P1, P2, disp12MaxDiff,preFilterCap,uniquenessRatio,speckleWindowSize,speckleRange,mode);
    stereoSGBM->compute(left_img, right_img, disp_img);
}

void apply_naive_block_matching(const cv::Mat& left_img, const cv::Mat& right_img, cv::Mat& disp_img, int block_size, int min_disparity, int max_disparity, float threshold) {
    int rad = floor(block_size / 2);

    unsigned int width = left_img.size().width;
    unsigned int height = left_img.size().height;

    for (int v = rad; v < height - rad; v++) {
        for (int u = rad; u < width - rad; u++) {
            auto x_patch = left_img(cv::Range(v-rad, v + rad + 1), cv::Range(u-rad, u + rad + 1));
            cv::Mat x = x_patch.clone().reshape(1, x_patch.total() * x_patch.channels()).t();

            cv::Mat y;
            cv::Mat concat_y;
            for (int d = min_disparity; d <= max_disparity; d++) {
                if (u - d < rad + 1) {
                    break;
                }

                auto y_patch = right_img(cv::Range(v-rad, v+rad+1), cv::Range(u-d-rad, u-d+rad+1));
                cv::Mat y_flatten = y_patch.clone().reshape(1, y_patch.total() * y_patch.channels()).t();

                if (y.empty()) {
                    y = y_flatten;
                } else {
                    cv::vconcat(y, y_flatten, concat_y);
                    y = concat_y;
                }
            }

            if (y.size().height > 0) {
                int h = y.size().height;
                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mat_x;
                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mat_y;
                cv::cv2eigen(x, mat_x);
                cv::cv2eigen(y, mat_y);

                std::vector<std::pair<float, int>> distPairs = pdist2SmallestK(mat_y, mat_x, 3);

                if (y.size().height > 1 && checkThreshold(distPairs, threshold)) {
                    disp_img.at<uint8_t>(v, u) = 0;
                } else {
                    disp_img.at<uint8_t>(v, u) = distPairs[0].second + min_disparity;
                }
            } else {
                disp_img.at<uint8_t>(v, u) = 0;
            }
        }
    }
}

/*
 * help functions implementation
 */
bool checkThreshold(std::vector<std::pair<float, int>> distPairs, float threshold) {
    float smallest = distPairs[0].first;
    bool isInvalid = true;
    for(int i = 0; i < distPairs.size(); i++) {
        if (distPairs[i].first / smallest >= threshold) {
            isInvalid = false;
            break;
        }
    }
    return isInvalid;
}



