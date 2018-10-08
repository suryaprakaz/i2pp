
// C++ STL
#include <iostream>

// OpenCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>


int main(int argc, char** argv) {

  // Arguments
  if(argc < 2) {
    std::cout << "Usage:" << std::endl;
    std::cout << "./rgb2grayscale [image]" << std::endl;
    return -1;
  }

  cv::Mat image;
  image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);

  // Invalid image
  if(!image.data) {
    std::cout << "Invalid image" << std::endl;
    return -1;
  }

  // Display original image
  cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
  cv::imshow("Original", image);
  cv::waitKey(0);
  return 0;
}
