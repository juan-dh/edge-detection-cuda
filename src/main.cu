#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <filesystem>
#include <cuda_runtime.h>
#include <npp.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <Exceptions.h>


__host__ FIBITMAP *loadImage(std::string sFilename)
{
    FREE_IMAGE_FORMAT eFormat = FreeImage_GetFileType(sFilename.c_str());
    FIBITMAP *pBitmap = FreeImage_Load(eFormat, sFilename.c_str());
    if (!pBitmap) {
        std::cerr << "Failed to load image: " << sFilename << std::endl;
        return nullptr;
    }
    return pBitmap;
}

__host__ FIBITMAP *convertToGray(FIBITMAP *pBitmap){
    FIBITMAP *pGray = FreeImage_ConvertToGreyscale(pBitmap);
    if (!pGray) {
        std::cerr << "Failed to convert image to grayscale." << std::endl;
        return nullptr;
    }
    return pGray;
}


__host__ void copyGrayBitmapToNPPImage(FIBITMAP *pGrayBitmap, npp::ImageCPU_8u_C1 &oHostScr)
{
    unsigned int w = FreeImage_GetWidth(pGrayBitmap);
    unsigned int h = FreeImage_GetHeight(pGrayBitmap);
    unsigned int nSrcPitch = FreeImage_GetPitch(pGrayBitmap);
    const Npp8u *pSrcLine = FreeImage_GetBits(pGrayBitmap) + nSrcPitch * (h - 1);
    Npp8u *pDstLine = oHostScr.data();
    unsigned int nDstPitch = oHostScr.pitch();
    for (unsigned int i = 0; i < h; ++i) {
        memcpy(pDstLine, pSrcLine, w);
        pSrcLine -= nSrcPitch;
        pDstLine += nDstPitch;
    }
}

__host__ void applyFilter(const npp::ImageNPP_8u_C1 &oDeviceSrc, npp::ImageNPP_8u_C1 &oDeviceDst, NppiSize oSizeROI, NppStreamContext &ctx, std::string edgeType)
{
    NppStatus status;
    
    if (edgeType == "horizontal") {
        status = nppiFilterSobelHoriz_8u_C1R_Ctx(
            oDeviceSrc.data(), oDeviceSrc.pitch(),
            oDeviceDst.data(), oDeviceDst.pitch(),
            oSizeROI, ctx);
    } else if (edgeType == "vertical") {
        status = nppiFilterSobelVert_8u_C1R_Ctx(
            oDeviceSrc.data(), oDeviceSrc.pitch(),
            oDeviceDst.data(), oDeviceDst.pitch(),
            oSizeROI, ctx);
    } else {
        std::cerr << "Unknown edge type: " << edgeType << std::endl;
        return;
    }
    
    if (status != NPP_SUCCESS) {
        std::cerr << "NPP error: " << status << std::endl;
    }
}

__host__ void parseArguments(int argc, char **argv, std::string &dataset, std::string &edgeType)
{
    dataset = "uscsipi";
    edgeType = "horizontal";
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            dataset = argv[++i];
            if (dataset != "stl10" && dataset != "uscsipi") {
                std::cerr << "Error: --dataset must be 'stl10' or 'uscsipi'" << std::endl;
                exit(1);
            }
        } else if (arg == "--edges" && i + 1 < argc) {
            edgeType = argv[++i];
            if (edgeType != "vertical" && edgeType != "horizontal") {
                std::cerr << "Error: --edges must be 'vertical' or 'horizontal'" << std::endl;
                exit(1);
            }
        }
    }
}

int main(int argc, char **argv)
{
    // Parse command line arguments
    std::string dataset;
    std::string edgeType;
    parseArguments(argc, argv, dataset, edgeType);
    
    std::cout << "Using dataset: " << dataset << std::endl;
    std::cout << "Using edge type: " << edgeType << std::endl;

    std::string inputDir = (dataset == "stl10") ? "data/stl10_images" : "data/uscsipi_images";


    // Clean directory and create output directory if it doesn't exist
    int i = 0;
    std::filesystem::remove_all("data/outputs");
    std::filesystem::create_directory("data/outputs");

    std::vector<std::filesystem::path> files;

    for (const auto &entry : std::filesystem::directory_iterator(inputDir)) {
        files.push_back(entry.path());
    }

    std::sort(files.begin(), files.end());

    for (const auto &path : files) {
        std::string sFilename = path.string();

        // Load the bitmap using FreeImage
        FIBITMAP *pBitmap = loadImage(sFilename);

        // Convert the bitmap to grayscale using FreeImage
        FIBITMAP *pGrayBitmap = convertToGray(pBitmap);

        // Create host NPP Images
        unsigned int w = FreeImage_GetWidth(pGrayBitmap);
        unsigned int h = FreeImage_GetHeight(pGrayBitmap);
        npp::ImageCPU_8u_C1 oHostScr;
        oHostScr = npp::ImageCPU_8u_C1(w, h);
        npp::ImageCPU_8u_C1 oHostDst;
        oHostDst = npp::ImageCPU_8u_C1(w, h);

        // Copy the grayscale bitmap to the NPP image
        copyGrayBitmapToNPPImage(pGrayBitmap, oHostScr);

        // Unload the FreeImage bitmaps
        FreeImage_Unload(pGrayBitmap);
        FreeImage_Unload(pBitmap);
        
        // Create device NPP Images and ROI 
        npp::ImageNPP_8u_C1 oDeviceSrc(oHostScr);
        npp::ImageNPP_8u_C1 oDeviceDst(oHostDst);
        NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};

        // Create NPP stream context and set it up with the current CUDA stream
        NppStreamContext ctx = {0};

        // Apply the Sobel filter using NPP
        applyFilter(oDeviceSrc, oDeviceDst, oSizeROI, ctx, edgeType);

        // Copy the result back to the host
        oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

        // Save the result as a PGM image
        std::string sResultFilename = std::string("data/outputs/") + dataset + "_" + edgeType + "_" + "img_" + std::to_string(i) + ".pgm";
        saveImage(sResultFilename, oHostDst);
        std::cout << "Saved image: " << sResultFilename << std::endl;

        i++;

    }

    return 0;
}
