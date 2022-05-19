#include <iostream>
#include <tiffio.h>

int main(int argc, char **argv){
    TIFF* tif = TIFFOpen("/home/leshinka/dev/ssb/Archive/3.tif46.tif116.tif105.tif102.tif", "r");
    // #define uint32 unsigned long
    unsigned int width, height;
    size_t npixels;
    unsigned int* raster;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);           // uint32 width;
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);        // uint32 height;
    npixels = width * height;
    raster = (unsigned int *) _TIFFmalloc(npixels * sizeof(unsigned int));
    TIFFReadRGBAImage(tif, width, height, raster, 0); 
    std::cout << raster[0] << std::endl;

    _TIFFfree(raster);
    TIFFClose(tif);
    return 0;
}
