
// C++ STL
#include <iostream>

// OpenCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>

// CUDA
#include <cuda.h>


__global__ void rgb2gray(const unsigned char* const rgbImage,
                         unsigned char* const greyImage, 
                         const int size) {

  for(int oIdx = blockIdx.x*blockDim.x + threadIdx.x;
      oIdx < size;
      oIdx += blockDim.x*gridDim.x)
  {
    int iIdx = 3 * oIdx;
    greyImage[oIdx] = (unsigned char) .114f * rgbImage[iIdx+0] + .587f * rgbImage[iIdx+1] + .299f * rgbImage[iIdx+2];
  }
}

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

  cv::Mat output(height, width, CV_8UC1, cv::Scalar(0));

  // Display original image
  cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
  cv::imshow("Original", image);
  cv::waitKey(0);

  cudaError_t error;

  unsigned char* rgb_image;
  unsigned char* gray_image;

  // device rgb input image allocator
  error = cudaMalloc((void **) &rgb_image, height*width*channels*sizeof(unsigned char));

  // device grayscale output image allocator
  error = cudaMalloc((void **) &gray_image, height*width*sizeof(unsigned char));

  if(error != cudaSuccess) 
  {
    printf("cudaMalloc returned error %s (code %d) (line %d)\n", cudaGetErrorString(error), error, __LINE__);
  }

  // Copy host image buffer to device
  cudaMemcpy(rgb_image, image.data, height*width*channels*sizeof(unsigned char), cudaMemcpyHostToDevice);

  rgb2gray<<<(int)(height*width)/1024,1024>>>(rgb_image, 
		    gray_image,
		    height*width);
                    
  // Copy grayscale device buffer to host
  cudaMemcpy(output.data, gray_image, height*width*sizeof(unsigned char), cudaMemcpyDeviceToHost);

  // Display grayscale converted image
  cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
  cv::imshow("Original", output);
  cv::waitKey(0);

  cudaFree(rgb_image);
  return 0;
}
