#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <tiffio.h>
#include <vector>

int main(int argc, char **argv){
    TIFF* tif = TIFFOpen("Archive/3.tif46.tif116.tif105.tif102.tif", "r");
    // #define uint32 unsigned long
    uint32 width, height;
    size_t npixels;
    uint32* raster;
    uint32 arrData[801][451];
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    npixels = width * height;
    raster = (unsigned int *) _TIFFmalloc(npixels * sizeof(unsigned int));
    TIFFReadRGBAImage(tif, width, height, raster, 0);
    std::cout << "width : " << width << std::endl << "height: "<< height << std::endl;
    for (int i= 0; i < width; i++){
        for (int j = 0; j < height; j++){
            int idx;
            idx = i * height + j;
            uint32 R=(uint32 )TIFFGetR(raster[idx]);
            uint32 G=(uint32 )TIFFGetG(raster[idx]);
            uint32 B=(uint32 )TIFFGetB(raster[idx]);
            std::cout << R << G << B << std::endl;
            if (R == 198 && G == 118 && B == 255){
                arrData[i][j] = 2;
            } else if (R == 79 && G == 255 && B == 130){
                arrData[i][j] = 1;
            } else if (R == 255 && G == 0 && B == 0){
                arrData[i][j] = 0;
            } else {

            }
        }
    }

    _TIFFfree(raster);
    TIFFClose(tif);
    std::cout << arrData[800][450] << std::endl;
    return 0;
}
