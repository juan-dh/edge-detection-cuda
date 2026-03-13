#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <npp.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <Exceptions.h>



int main(int argc, char **argv)
{
    try {
        std::string sFilename = "data/images/img_0000.jpg"; 
        npp::ImageCPU_8u_C1 oHostScr;
        // the helper in ImageIO only supports single‑channel files; if the
        // input happens to be color we convert it manually to gray before
        // wrapping it in an ImageCPU object.
        
        {
            // attempt direct load first; this will assert if the file is
            // not 8‑bit greyscale
            try {
                npp::loadImage(sFilename, oHostScr);
            }
            catch (const npp::Exception &e) {
                // assume failure due to non‑grayscale; fall back to manual
                // FreeImage conversion
                FREE_IMAGE_FORMAT eFormat = FreeImage_GetFileType(sFilename.c_str());
                if (eFormat == FIF_UNKNOWN)
                    eFormat = FreeImage_GetFIFFromFilename(sFilename.c_str());
                FIBITMAP *pBitmap = nullptr;
                if (FreeImage_FIFSupportsReading(eFormat))
                    pBitmap = FreeImage_Load(eFormat, sFilename.c_str());
                if (!pBitmap)
                    throw; // re‑throw original
                // convert to 8‑bit greyscale
                FIBITMAP *pGray = FreeImage_ConvertToGreyscale(pBitmap);
                FreeImage_Unload(pBitmap);
                if (!pGray)
                    throw npp::Exception("failed to convert to greyscale");
                // copy pixels into oHostScr
                unsigned int w = FreeImage_GetWidth(pGray);
                unsigned int h = FreeImage_GetHeight(pGray);
                oHostScr = npp::ImageCPU_8u_C1(w, h);
                unsigned int nSrcPitch = FreeImage_GetPitch(pGray);
                const Npp8u *pSrcLine = FreeImage_GetBits(pGray) + nSrcPitch * (h -1);
                Npp8u *pDstLine = oHostScr.data();
                unsigned int nDstPitch = oHostScr.pitch();
                for (unsigned int iLine = 0; iLine < h; ++iLine) {
                    memcpy(pDstLine, pSrcLine, w * sizeof(Npp8u));
                    pSrcLine -= nSrcPitch;
                    pDstLine += nDstPitch;
                }
                FreeImage_Unload(pGray);
            }
        }

        npp::ImageNPP_8u_C1 oDeviceSrc(oHostScr);

        NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};

        npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);

        NppStreamContext ctx = {0};

        NppStatus status = nppiFilterSobelHoriz_8u_C1R_Ctx(
            oDeviceSrc.data(), oDeviceSrc.pitch(),
            oDeviceDst.data(), oDeviceDst.pitch(),
            oSizeROI,
            ctx);

        if (status != NPP_SUCCESS)
            std::cout << "NPP error: " << status << std::endl;

        npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
        oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

        std::string sResultFilename = "data/outputs/sobel_img_0000.jpg"; 
        saveImage(sResultFilename, oHostDst);
        std::cout << "Saved image: " << sResultFilename << std::endl;
    }
    catch (const npp::Exception &e) {
        std::cerr << "NPP exception: " << e.toString() << "\n";
        return 1;
    }
    catch (const std::exception &e) {
        std::cerr << "std::exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
