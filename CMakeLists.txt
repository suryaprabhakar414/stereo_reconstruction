#
# Created by Zhijie Yang on 14.12.21.
#

cmake_minimum_required(VERSION 3.10)

project(stereo_reconstruction)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )
find_package(PCL 1.3 REQUIRED COMPONENTS common io visualization)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(PNG REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${PNG_INCLUDE_DIRS})

# Define header and source files
include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${PCL_LIBRARY_DIRS}
)

file(GLOB src_list
        "main.cpp"
        "eval.cpp"
        "src/*.cpp"
        )


add_definitions(${PCL_DEFINITIONS})

add_executable(${PROJECT_NAME} main.cpp src/disparity.cpp include/disparity.h include/pdist2.h include/types.h src/triangulation_indirect.cpp include/triangulation_indirect.h include/defs.h src/registration.cpp include/registration.h include/io_util.h include/matrix.h src/matrix.cpp include/pose_eval.h )
add_executable(${PROJECT_NAME}_eval eval.cpp include/disparity_eval.h include/io_disp.h include/io_integer.h include/io_util.h include/matrix.h src/matrix.cpp include/pose_eval.h )

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES}
        ${PCL_VISUALIZATION_LIBRARIES} pcl_visualization Eigen3::Eigen ${PNG_LIBRARY} -lpcl_filters)
target_link_libraries( ${PROJECT_NAME}_eval ${OpenCV_LIBS} ${PCL_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES}
        ${PCL_VISUALIZATION_LIBRARIES} pcl_visualization Eigen3::Eigen ${PNG_LIBRARY})
# add_dependencies(${PROJECT_NAME})
#target_link_libraries(${PROJECT_NAME} libSimpleMesh pthread)
# target_include_directories(interactive_arap PUBLIC ${Eigen3_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR})

