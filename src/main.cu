#include <iostream>
#include <string>
#include <filesystem>
#include <cuda_runtime.h>
#include <npp.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <Exceptions.h>

int main(int argc, char **argv)
{
    std::filesystem::create_directory("data/outputs");

    for (const auto& entry : std::filesystem::directory_iterator("data/images")) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
            std::string sFilename = entry.path().string();
            npp::ImageCPU_8u_C1 oHostScr;
            
            // Load image, converting to grayscale if needed
            try {
                npp::loadImage(sFilename, oHostScr);
            } catch (const npp::Exception &) {
                // Convert color image to grayscale
                FREE_IMAGE_FORMAT eFormat = FreeImage_GetFileType(sFilename.c_str());
                if (eFormat == FIF_UNKNOWN) eFormat = FreeImage_GetFIFFromFilename(sFilename.c_str());
                FIBITMAP *pBitmap = FreeImage_Load(eFormat, sFilename.c_str());
                FIBITMAP *pGray = FreeImage_ConvertToGreyscale(pBitmap);
                FreeImage_Unload(pBitmap);
                unsigned int w = FreeImage_GetWidth(pGray), h = FreeImage_GetHeight(pGray);
                oHostScr = npp::ImageCPU_8u_C1(w, h);
                unsigned int nSrcPitch = FreeImage_GetPitch(pGray);
                const Npp8u *pSrcLine = FreeImage_GetBits(pGray) + nSrcPitch * (h - 1);
                Npp8u *pDstLine = oHostScr.data();
                unsigned int nDstPitch = oHostScr.pitch();
                for (unsigned int i = 0; i < h; ++i) {
                    memcpy(pDstLine, pSrcLine, w);
                    pSrcLine -= nSrcPitch;
                    pDstLine += nDstPitch;
                }
                FreeImage_Unload(pGray);
            }

            npp::ImageNPP_8u_C1 oDeviceSrc(oHostScr);
            NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
            npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);

            NppStreamContext ctx = {0};
            NppStatus status = nppiFilterSobelHoriz_8u_C1R_Ctx(
                oDeviceSrc.data(), oDeviceSrc.pitch(),
                oDeviceDst.data(), oDeviceDst.pitch(),
                oSizeROI, ctx);

            if (status != NPP_SUCCESS) std::cout << "NPP error: " << status << std::endl;

            npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
            oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

            std::string sResultFilename = "data/outputs/sobel_" + entry.path().stem().string() + ".pgm";
            saveImage(sResultFilename, oHostDst);
            std::cout << "Saved image: " << sResultFilename << std::endl;
        }
    }

    return 0;
}
