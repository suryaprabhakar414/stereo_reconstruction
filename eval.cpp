#include <iostream>
#include "disparity_eval.h"
#include "io_util.h"
#include "pose_eval.h"
#include "matrix.h"

int main(int argc, char** argv) {
    // Define evaluate on disparity map or pose, default pose
    std::string eval_type = argv[1]? argv[1]: "POSE";

    std::transform(eval_type.begin(), eval_type.end(),eval_type.begin(), ::toupper);
    if (eval_type != "DISPARITY" && eval_type != "POSE") {
        std::cout<< "Invalid evaluation type input: " << eval_type << ", please choose from DISPARITY, POSE." << std::endl;
        return 1;
    }

    if (eval_type == "DISPARITY") {
        // Disparity Map Evaluation
        for (int i = 0; i < 5; i++) {
            disparity_eval(getAlgorithmFromIndex(i)); // iterate and evaluate all matching algorithms
        }
    } else {
        // Pose Evaluation
        for (int i = 0; i < 5; i++) {
            std::string algo_name = getAlgorithmFromIndex(i);
            if (isDirectoryExist((std::string("../result/") + algo_name).c_str())) {
                pose_eval(algo_name); // iterate and evaluate all matching algorithms
            }
        }
    }
}
