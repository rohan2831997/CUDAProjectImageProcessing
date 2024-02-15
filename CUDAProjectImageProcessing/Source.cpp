// Heavily Borrowed from CUDA SAMPLES From NVIDA




#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64) 
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include <npp.h>
#include <string.h>

#include <iostream>
#include <fstream>

inline int cudaDeviceInit(int argc, const char** argv) {
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "CUDA Error : No CUDA capable devices detected." << std::endl;
        exit(EXIT_FAILURE);
    }

    int dev = findCudaDevice(argc, argv);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name
        << std::endl;

    checkCudaErrors(cudaSetDevice(dev));
}

bool printfNPPinfo(int argc, char* argv[])
{
    const NppLibraryVersion* libVer = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("    CUDA Driver version : %d.%d\n", driverVersion / 1000,
        (driverVersion % 100) / 10);
    printf("    CUDA Runtime version : %d.%d\n", runtimeVersion / 1000,
        (runtimeVersion % 100) / 10);

    return checkCudaCapabilities(1, 0);
}


int main(int argc, char* argv[])
{
    try {
        std::string imgPath = "data/teapot512.pgm";
        std::string outDirectory = "";

        if (printfNPPinfo(argc, argv) == false) {
            cudaDeviceReset();
            exit (EXIT_SUCCESS);
        }

        cudaDeviceInit(argc, (const char**)argv);

        // Skipping the part to take different Files as of now

        char* filePath;

        if (checkCmdLineFlag(argc, (const char**)argv, "input")) {
            getCmdLineArgumentString(argc, (const char**)argv, "input", &filePath);
            imgPath = filePath;
            imgPath.insert(0, "data/");
            
        }


        std::ifstream imgFile(imgPath.data(), std::ifstream::in);

        if (imgFile.good()) {
            std::cout << imgPath << " Successully Opened" << std::endl;
            imgFile.close();
        }
        else
        {
            std::cout << imgPath << " Could not be opened" << std::endl;
            imgFile.close();
            cudaDeviceReset();
            exit(EXIT_FAILURE);
        }

        // Name the output file Directory
        std::string outputPath;
        outputPath = imgPath;

        std::string::size_type dot = outputPath.rfind('.');

        if (dot != std::string::npos) {
            outputPath = outputPath.substr(0, dot);
        }

        std::string outputx = outputPath + "output.pgm";
        std::string outputy = outputPath + "outputy.pgm";

        // 8 bit greyscale image
        npp::ImageCPU_8u_C1 oHostSrc;

        // Load Image
        npp:loadImage(imgPath, oHostSrc);

        // upload from Host to the device
        npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

        NppiSize oSrcSize = { (int)oDeviceSrc.width(), (int)oDeviceSrc.height() };
        NppiPoint oSrcOffset = { 0, 0 };
        
        // Decalre our region of interest
        NppiSize oSizeROI = { (int)oDeviceSrc.width(), (int)oDeviceSrc.height() };
        
        //Allocate device Destination images of appropriate sizes
        npp::ImageNPP_16s_C1 oDeviceDstX(oSizeROI.width, oSizeROI.height);
        npp::ImageNPP_16s_C1 oDeviceDstY(oSizeROI.width, oSizeROI.height);

        NPP_CHECK_NPP(nppiGradientVectorPrewittBorder_8u16s_C1R(
            oDeviceSrc.data(), oDeviceSrc.pitch(), oSrcSize, oSrcOffset,
            oDeviceDstX.data(), oDeviceDstX.pitch(), oDeviceDstY.data(),
            oDeviceDstY.pitch(), 0, 0, 0, 0, oSizeROI, NPP_MASK_SIZE_3_X_3,
            nppiNormL1, NPP_BORDER_REPLICATE));

        //allocate device destitaion images of appropriatedely size
        npp::ImageNPP_8u_C1 oDeviceDSTOutX(oSizeROI.width, oSizeROI.height);
        npp::ImageNPP_8u_C1 oDeviceDSTOutY(oSizeROI.width, oSizeROI.height);

        //convert 16s_C1 result images to binary 8u_C1 output images using constant
        // value 
        NPP_CHECK_NPP(nppiCompareC_16s_C1R(
            oDeviceDstX.data(), oDeviceDstX.pitch(), 32, oDeviceDSTOutX.data(), oDeviceDSTOutX.pitch(),
            oSizeROI, NPP_CMP_GREATER_EQ
        ));

        NPP_CHECK_NPP(nppiCompareC_16s_C1R(
            oDeviceDstY.data(), oDeviceDstY.pitch(), 32, oDeviceDSTOutY.data(), oDeviceDSTOutY.pitch(),
            oSizeROI, NPP_CMP_GREATER_EQ
        ));
        //copy device result data into them
        
        // Create Host Images for the results
        npp::ImageCPU_8u_C1 oHostDstX(oDeviceDSTOutX.size());
        npp::ImageCPU_8u_C1 oHostDstY(oDeviceDSTOutY.size());

        //Copy the results
        oDeviceDSTOutX.copyTo(oHostDstX.data(), oHostDstX.pitch());
        oDeviceDSTOutX.copyTo(oHostDstY.data(), oHostDstY.pitch());

        saveImage(outputx, oHostDstX);
        std::cout << "Saved image: " << outputx << std::endl;
        //saveImage(outputy, oHostDstY);
        //std::cout << "Saved image: " << outputy << std::endl;
    }
    catch (...)
    {

    }
	return 0;
}

