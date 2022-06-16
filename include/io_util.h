//
// Created by Ryou on 2022/2/2.
//
#include <sys/stat.h>
#include "defs.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#ifndef STEREO_RECONSTRUCTION_IO_UTIL_H
#define STEREO_RECONSTRUCTION_IO_UTIL_H

using namespace std;

bool isDirectoryExist(const char *dir) {
    struct stat info;
    if(stat(dir, &info) != 0)
        return false;
    else if(info.st_mode & S_IFDIR)
        return true;
    else
        return false;
}

void prepareDirectory(std::string dir) {
    int len = dir.length();
    char tempDirPath[256] = {};
    for (int i = 0; i < len; i++) {
        tempDirPath[i] = dir[i];
        if (tempDirPath[i] == '/') {
            if (!isDirectoryExist(tempDirPath)) {
                mkdir(tempDirPath, 0777);
            }
        }
    }
}

// Save outputs for further evaluation
void savePoses(const std::string& dir, const std::string& name, std::vector<Eigen::Matrix4f> transforms) {
    prepareDirectory(dir);

    ofstream output;
    output.open(dir + "/" + name + ".txt", ios::trunc);

    for(int i = 0; i < transforms.size(); i++) {
        Eigen::Matrix4f transform = transforms[i];
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                if (j == 3 and k == 3) {
                    output << transform(j, k) ;
                } else {
                    output << transform(j, k) << ", ";
                }
            }
        }
        output << "\n";
    }
    output.close();
}

// Save outputs for disparity map
void saveDisparityMap(std::string dir, std::string image_path, cv::Mat disparity) {
    prepareDirectory(dir);

    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);
    compression_params.push_back(cv::IMWRITE_PNG_STRATEGY);
    compression_params.push_back(cv::IMWRITE_PNG_STRATEGY_DEFAULT);

    std::string path = dir + image_path;
    cv::imwrite(path, disparity, compression_params);
}

int checkInputAlgorithmValid(std::string algorithm) {
    if (algorithm == "SURF") {
        return Algorithm(SURF);
    } else if (algorithm == "SIFT") {
        return Algorithm(SIFT);
    } else if (algorithm == "ORB") {
        return Algorithm(ORB);
    } else if (algorithm == "BM") {
        return Algorithm(BM);
    } else if (algorithm == "SGBM") {
        return Algorithm(SGBM);
    } else {
        return -1;
    }
}

std::string getAlgorithmFromIndex(int algorithm) {
    if (algorithm == 0) {
        return "SURF";
    } else if (algorithm == 1) {
        return "SIFT";
    } else if (algorithm == 2) {
        return "ORB";
    } else if (algorithm == 3) {
        return "BM";
    } else if (algorithm == 4) {
        return "SGBM";
    }
}

#endif //STEREO_RECONSTRUCTION_IO_UTIL_H
