#include <iostream>
#include <tiffio.h>

int main(int argc, char **argv){
    TIFF* tif = TIFFOpen("Archive/3.tif46.tif116.tif105.tif102.tif", "r");
    // #define uint32 unsigned long
    uint32 width, height;
    size_t npixels;
    uint32* raster;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    npixels = width * height;
    raster = (unsigned int *) _TIFFmalloc(npixels * sizeof(unsigned int));
    TIFFReadRGBAImage(tif, width, height, raster, 0);
    std::cout << "width : " << width << std::endl << "height: "<< height << std::endl; 
    std::cout << raster[100] << std::endl;

    _TIFFfree(raster);
    TIFFClose(tif);
    return 0;
}
