#ifndef STEREO_RECONSTRUCTION_DISPARITY_EVAL_H
#define STEREO_RECONSTRUCTION_DISPARITY_EVAL_H

#include <string>
#include <cstdint>
#include <iostream>
#include <vector>
#include "io_disp.h"
#include "io_integer.h"

// Number of result images for evaluation
//#define NUM_TEST_IMAGES 200
#define NUM_TEST_IMAGES 100

//#define NUM_ERROR_IMAGES 20
#define NUM_ERROR_IMAGES 100

#define ABS_THRESH 3.0
#define REL_THRESH 0.05

// Check whether the images in the given directory are larger than images needed(NUM_TEST_IMAGES) for evaluation
int resultsAvailable(std::string dir) {
    std::int32_t count = 0;
    for (std::int32_t i = 0; i < NUM_TEST_IMAGES; i++) {
        char prefix[256];

        // The format of image name is for example, 000001_10.png
        sprintf(prefix, "%06d_10", i);
        FILE *tmp_file = fopen((dir + "/" + prefix + ".png").c_str(), "rb");
        if (tmp_file) {
            count++;
            fclose(tmp_file);
        }
    }

    return count;
}

// copied from utils.h
bool imageFormat(std::string file_name, png::color_type col, size_t depth, int32_t width, int32_t height) {
    std::ifstream file_stream;
    file_stream.open(file_name.c_str(), std::ios::binary);
    png::reader <std::istream> reader(file_stream);
    reader.read_info();
    if (reader.get_color_type() != col) return false;
    if (reader.get_bit_depth() != depth) return false;
    if (reader.get_width() != width) return false;
    if (reader.get_height() != height) return false;
    return true;
}

std::vector<float> disparityErrorsOutlier(DisparityImage &D_gt, DisparityImage &D_orig, DisparityImage &D_ipol, IntegerImage &O_map) {

    // check file size
    if (D_gt.width() != D_orig.width() || D_gt.height() != D_orig.height()) {
        std::cout << "ERROR: Wrong file size!" << std::endl;
        throw 1;
    }

    // extract width and height
    int32_t width = D_gt.width();
    int32_t height = D_gt.height();

    // init errors
    std::vector<float> errors;
    int32_t num_errors_bg = 0;
    int32_t num_pixels_bg = 0;
    int32_t num_errors_bg_result = 0;
    int32_t num_pixels_bg_result = 0;
    int32_t num_errors_fg = 0;
    int32_t num_pixels_fg = 0;
    int32_t num_errors_fg_result = 0;
    int32_t num_pixels_fg_result = 0;
    int32_t num_errors_all = 0;
    int32_t num_pixels_all = 0;
    int32_t num_errors_all_result = 0;
    int32_t num_pixels_all_result = 0;

    // for all pixels do
    for (int32_t u = 0; u < width; u++) {
        for (int32_t v = 0; v < height; v++) {
            if (D_gt.isValid(u, v)) {
                float d_gt = D_gt.getDisp(u, v);
                float d_est = D_ipol.getDisp(u, v);
                bool d_err = fabs(d_gt - d_est) > ABS_THRESH && fabs(d_gt - d_est) / fabs(d_gt) > REL_THRESH;
                if (O_map.getValue(u, v) == 0) {
                    if (d_err)
                        num_errors_bg++;
                    num_pixels_bg++;
                    if (D_orig.isValid(u, v)) {
                        if (d_err)
                            num_errors_bg_result++;
                        num_pixels_bg_result++;
                    }
                } else {
                    if (d_err)
                        num_errors_fg++;
                    num_pixels_fg++;
                    if (D_orig.isValid(u, v)) {
                        if (d_err)
                            num_errors_fg_result++;
                        num_pixels_fg_result++;
                    }
                }
                if (d_err)
                    num_errors_all++;
                num_pixels_all++;
                if (D_orig.isValid(u, v)) {
                    if (d_err)
                        num_errors_all_result++;
                    num_pixels_all_result++;
                }
            }
        }
    }

    // push back errors and pixel count
    errors.push_back(num_errors_bg);
    errors.push_back(num_pixels_bg);
    errors.push_back(num_errors_bg_result);
    errors.push_back(num_pixels_bg_result);
    errors.push_back(num_errors_fg);
    errors.push_back(num_pixels_fg);
    errors.push_back(num_errors_fg_result);
    errors.push_back(num_pixels_fg_result);
    errors.push_back(num_errors_all);
    errors.push_back(num_pixels_all);
    errors.push_back(num_errors_all_result);
    errors.push_back(num_pixels_all_result);

    // push back density
    errors.push_back((float) num_pixels_all_result / std::max((float) num_pixels_all, 1.0f));

    // return errors
    return errors;
}

bool disparity_eval(std::string algorithm) {
    std::string dir = "../result/" + algorithm;
    std::string gt_obj_map_dir = "../images/stereo/obj_map";
    std::string gt_disp_noc_0_dir = "../images/stereo/disp_noc_0";
    std::string gt_disp_occ_0_dir = "../images/stereo/disp_occ_0";
    std::string result_disp_0_dir = dir + "/data/disp_0";

    int count = resultsAvailable(result_disp_0_dir);

    int errorCount = count; // In original setting, NUM_TEST_IMAGES is 200 and NUM_ERROR_IMAGES is 20

    if (count != 0) {
        // create output directories (depending on which benchmarks to evaluate)
        system(("mkdir " + dir + "/errors_disp_noc_0/").c_str());
        system(("mkdir " + dir + "/errors_disp_occ_0/").c_str());
        system(("mkdir " + dir + "/errors_disp_img_0/").c_str());
        system(("mkdir " + dir + "/result_disp_img_0/").c_str());

        // accumulators
        float errors_disp_noc_0[3 * 4] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        float errors_disp_occ_0[3 * 4] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        // for all test files do
        for (int32_t i = 0; i < count; i++) {
            // file name
            char prefix[256];
            sprintf(prefix, "%06d_10", i);

            std::cout << "Test file: " << std::string(prefix) << std::endl;

            try {
                DisparityImage D_gt_noc_0, D_gt_occ_0, D_orig_0, D_ipol_0;
                DisparityImage D_gt_noc_1, D_gt_occ_1, D_orig_1, D_ipol_1;

                // load object map (0:background, >0:foreground)
                IntegerImage O_map = IntegerImage(gt_obj_map_dir + "/" + prefix + ".png");

                // load ground truth disparity maps
                D_gt_noc_0 = DisparityImage(gt_disp_noc_0_dir + "/" + prefix + ".png");
                D_gt_occ_0 = DisparityImage(gt_disp_occ_0_dir + "/" + prefix + ".png");

                // check submitted result
                std::string image_file = result_disp_0_dir + "/" + prefix + ".png";
                if (!imageFormat(image_file, png::color_type_gray, 16, D_gt_noc_0.width(), D_gt_noc_0.height())) {
                    std::cout << "ERROR: Input must be png, 1 channel, 16 bit" << D_gt_noc_0.width() << "x"
                              << D_gt_noc_0.height() << "px" << std::endl;
                    return false;
                }

                // load submitted result and interpolate missing values
                D_orig_0 = DisparityImage(image_file);
                D_ipol_0 = DisparityImage(D_orig_0);
                D_ipol_0.interpolateBackground();

                // calculate disparity errors
                std::vector<float> errors_noc_curr = disparityErrorsOutlier(D_gt_noc_0, D_orig_0, D_ipol_0, O_map);
                std::vector<float> errors_occ_curr = disparityErrorsOutlier(D_gt_occ_0, D_orig_0, D_ipol_0, O_map);

                // accumulate errors
                for (int32_t j = 0; j < errors_noc_curr.size() - 1; j++) {
                    errors_disp_noc_0[j] += errors_noc_curr[j];
                    errors_disp_occ_0[j] += errors_occ_curr[j];
                }

                // save error images
                if (i < errorCount) {

                    // save errors of error images to text file
                    FILE *errors_noc_file = fopen((dir + "/errors_disp_noc_0/" + prefix + ".txt").c_str(), "w");
                    FILE *errors_occ_file = fopen((dir + "/errors_disp_occ_0/" + prefix + ".txt").c_str(), "w");
                    for (int32_t i = 0; i < 12; i += 2)
                        fprintf(errors_noc_file, "%f ", errors_noc_curr[i] / std::max(errors_noc_curr[i + 1], 1.0f));
                    fprintf(errors_noc_file, "%f ", errors_noc_curr[12]);
                    for (int32_t i = 0; i < 12; i += 2)
                        fprintf(errors_occ_file, "%f ", errors_occ_curr[i] / std::max(errors_occ_curr[i + 1], 1.0f));
                    fprintf(errors_occ_file, "%f ", errors_occ_curr[12]);
                    fclose(errors_noc_file);
                    fclose(errors_occ_file);

                    // save error image
                    D_ipol_0.errorImage(D_gt_noc_0, D_gt_occ_0, true).write(
                            dir + "/errors_disp_img_0/" + prefix + ".png");

                    // compute maximum disparity
                    float max_disp = D_gt_occ_0.maxDisp();

                    // save interpolated disparity image false color coded
                    D_ipol_0.writeColor(dir + "/result_disp_img_0/" + prefix + ".png", max_disp);
                }
            } catch (...) {
                std::cout << "ERROR: Couldn't read:" << prefix << ".png" << std::endl;
                return false;
            }
        }
        std::cout << "Evaluation Done" << std::endl;
    } else {
        std::cout << "Not enough result images found for any of the evaluations, stopping evaluation." << std::endl;
    }
    return 0;
}

#endif //STEREO_RECONSTRUCTION_DISPARITY_EVAL_H
