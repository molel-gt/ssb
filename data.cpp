#include <iostream>
#include <tiffio.h>
#include <vector>

int main(int argc, char **argv){
    TIFF* tif = TIFFOpen("Archive/3.tif46.tif116.tif105.tif102.tif", "r");
    uint32 width, length;
    size_t npixels;
    uint32* raster;
    uint32 arrData[451][800];
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &length);
    npixels = width * length;
    raster = (unsigned int *) _TIFFmalloc(npixels * sizeof(unsigned int));
    TIFFReadRGBAImage(tif, width, length, raster, 0);
    std::cout << "length: " << length << std::endl << "width : "<< width << std::endl;
    for (int i= 0; i < length; i++){
        for (int j = 0; j < width; j++){
            int idx;
            idx = i + j * length;
            uint32 R = (uint32 )TIFFGetR(raster[idx]);
            uint32 G = (uint32 )TIFFGetG(raster[idx]);
            uint32 B = (uint32 )TIFFGetB(raster[idx]);
            // std::cout << R << ", " << G << ", "  << B << std::endl;
            if ((B == 255) && (G == 118) && (R == 198)){
                arrData[i][j] = 2;
            } else if ((B == 130) && (G == 255) && (R == 79)){
                arrData[i][j] = 1;
            } else if ((B == 0) && (G == 0) && (R == 255)){
                arrData[i][j] = 0;
            } else {

            }
        }
    }

    _TIFFfree(raster);
    TIFFClose(tif);
    std::cout << arrData[450][800] << std::endl;
    return 0;
}
