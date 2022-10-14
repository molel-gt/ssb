/*
  mexFunction to calculate a histogram of m-cubes
   Input : binary image volume
   Output: number of occurences of each m-cube type in the volume

   Copyright (C) 2004-2010 Joakim Lindblad
    Not for distribution! Newer, public files will
    be available at http://www.cb.uu.se/~joakim/software/
*/
#include <mex.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   /* Check for proper input and output */    
   if (nrhs != 1) {
      mexErrMsgTxt("One input argument required.");
   } 
   if(nlhs > 1){
      mexErrMsgTxt("Too many output arguments.");
   }
   if (!(mxIsLogical(prhs[0]))) {
      mexErrMsgTxt("Input must be of type logical.");
   }
   if (mxGetNumberOfDimensions(prhs[0])!=3) {
      mexErrMsgTxt("Input must be three dimensional.");
   }

   const mwSize* dims=mxGetDimensions(prhs[0]);

   /* Create output matrix initialized to 0 */
   plhs[0] = mxCreateDoubleMatrix(256,1,mxREAL); 

   /* Pointer to output histogram */
   double* m_cubes=mxGetPr(plhs[0]); 

   mwSize line_size=dims[0];
   mwSize plane_size=dims[0]*dims[1];

   /* Pointer to image data */
   const mxLogical* p=mxGetLogicals(prhs[0]); 
   /* Skip right/down/far side */
   mwSize x,y,z;
   unsigned char code;
   for (z=0;z<dims[2]-1;++z) {
      for (y=0;y<dims[1]-1;++y) {
         for (x=0,code=0;x<dims[0];++x,++p,code>>=4) {
            if (*p) code |= 0x10;
            if (*(p+line_size)) code |= 0x20;
            if (*(p+plane_size)) code |= 0x40;
            if (*(p+plane_size+line_size)) code |= 0x80;
            if (x>0) m_cubes[code]++;
         }
      }
      p+=line_size; /* the missed edge */
   }
}
