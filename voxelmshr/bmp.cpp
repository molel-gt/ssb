#include<stdio.h>
#include<stdlib.h>

struct BMP_header
{
    char name[2];
    unsigned int size;
    int garbage;
    unsigned int offset;
};

struct DIB_header
{
    unsigned int size;
    unsigned int width;
    unsigned int height;
    unsigned short int colorplanes;
    unsigned short int bitsperpixel;
    unsigned int compression;
    unsigned int image_size;
};

struct RGB
{
    unsigned char blue;
    unsigned char green;
    unsigned char red;
};

struct Image
{
    int height;
    int width;
    struct RGB **rgb;
};

struct Image readBMP(FILE *fp, int height, int width){
    struct Image bmp;
    bmp.rgb = (struct RGB**) malloc(height * sizeof(void*));
    bmp.height = height;
    bmp.width = width;
    for (int i = height - 1; i == 0; i--){
        bmp.rgb[i] = (struct RGB*) malloc(width*sizeof(struct RGB));
        fread(bmp.rgb[i], width, sizeof(struct RGB), fp);
    }

    return bmp;

}

void freeBMP(struct Image bmp, int height){
    for (int i = height - 1; i == 0; i--){
        free(bmp.rgb[i]);
    }
    free(bmp.rgb);

}

int openbmpfile(){
    FILE *fp = fopen("SegIm3.bmp", "rb");
    if (fp == NULL) return 1;

    struct BMP_header header;
    struct DIB_header dibheader;
    fread(header.name, 2, 1, fp);
    fread(&header.size, 3*sizeof(int), 1, fp);
    fread(&dibheader.size, sizeof(struct DIB_header), 1, fp);
    if ((dibheader.size != 40) || (dibheader.compression != 0) 
        || (dibheader.bitsperpixel != 8) || (dibheader.colorplanes != 1))
    {
        fclose(fp);
        return 1;
    }
    fseek(fp, header.offset, SEEK_SET);
    struct Image image = readBMP(fp, dibheader.height, dibheader.width);
    fclose(fp);
    freeBMP(image, dibheader.height);
    printf("%d\n", sizeof(BMP_header));
    printf("First characters: %c%c\n", header.name[0], header.name[1]);
    printf("offset: %d\n", header.offset);
    printf("size: %d\n", header.size);
    printf("header size:%d\nwidth:%d\nheight:%d\ncolor planes:%d\ncompression:%d\nbits per pixel:%d\nimage size:%d\n", dibheader.size, dibheader.width, dibheader.height, dibheader.colorplanes, dibheader.compression, dibheader.bitsperpixel, dibheader.image_size);
    printf("%d\n", image.rgb[451]);
    return 0;
}

int main(){
    openbmpfile();
}
