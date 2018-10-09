
// C++ STL
#include <iostream>

// OpenCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>

// CUDA
#include <cuda.h>


int main(int argc, char** argv) {

  // Arguments
  if(argc < 2) {
    std::cout << "Usage:" << std::endl;
    std::cout << "\t./rgb2grayscale [image]" << std::endl;
    return -1;
  }

  cv::Mat image;
  image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);

  // Invalid image
  if(!image.data) {
    std::cout << "Invalid image" << std::endl;
    return -1;
  }

  int height   = image.rows;
  int width    = image.cols;
  int channels = image.channels();

  std::cout << "Height: "   << height   << std::endl;
  std::cout << "Width: "    << width    << std::endl;
  std::cout << "Channels: " << channels << std::endl;

  // Display original image
  cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
  cv::imshow("Original", image);
  cv::waitKey(0);

  // Declare and allocate uchar4 host and device memory buffers
  uchar4 *h_image;
  uchar4 *d_image;

  cudaError_t error;

  // uchar4 host image allocator
  h_image = (uchar4 *) malloc(height*width*sizeof(uchar4));

  if(!h_image)
  {
    std::cout << "Unable to allocate memory" << std::endl;
  }

  // uchar4 device image allocator
  error = cudaMalloc((void **) &d_image, height*width*sizeof(uchar4));

  if(error != cudaSuccess) 
  {
    printf("cudaMalloc returned error %s (code %d) (line %d)\n", cudaGetErrorString(error), error, __LINE__);
  }

  for(size_t i=0; i<height*width; i++) {
    unsigned char *pixel = image.data + 3*i;
    h_image[i]           = make_uchar4(pixel[0], pixel[1], pixel[2], 0);
  }
  // Copy host image buffer to device
  cudaMemcpy(d_image, h_image, height*width*sizeof(uchar4), cudaMemcpyHostToDevice);

  free(h_image);
  cudaFree(d_image);
  return 0;
}
