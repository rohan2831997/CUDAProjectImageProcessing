# Edge Detector using CUDA

I have implemented edge detection(in x and y direction) using GPU Acceleration (CUDA)

## Project Organisation

1. There is Visual Studio solution (.sln) file which can be used to see the source code.
2. All the dependencies are present in CUDAProjectImageProcessing/include
3. The Images we would like to edit must be present in CUDAProjectImageProcessing/data

## Running the program
1. The exe file is present in x64/debug/CUDAProjectImageProcessing.exe
2. To run the program with the image ("sample.pgm"), the file must be present in x64/debug/data,
Run CMD from the location of the program and run the following command. 

```bash
CUDAProjectImageProcessing.exe input sample.pgm
```
3. The Outputs will get saved in x64/debug/data with (imageRootName)outputx.pgm and (imageRootName)outputy.pgm

## Acknowledgments and Credits
1. This Project is part of the coursera course : [CUDA at Scale for the Enterprise](https://www.coursera.org/learn/cuda-at-scale-for-the-enterprise)
2. I have borrowed heavily from NVIDIA CUDA samples which can be obtained at [CUDA SAMPLES](https://github.com/NVIDIA/cuda-samples)
