// #include <complex.h>
#include <exception>
#include <fstream>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <iomanip>
#include <iostream>
#include <utility>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

struct Parameters {
    gsl_complex A;
    std::pair<double, double> coefs[];
    // double angles[]; // multiples of pi
    // double prevertices[];
};

gsl_complex dz_dt(gsl_complex t, void *p){
    struct Parameters *params = (struct Parameters *)p;
    int n = sizeof(params->coefs) / (sizeof(params->coefs[0].first));
    gsl_complex res = params->A;
    for (int i=0; i < n; i++){
        res = gsl_complex_mul(res,
                              gsl_complex_pow(
                                              gsl_complex_add_real(gsl_complex_div(-1* t, params->coefs[i][0]), 1),
                                              params->coefs[i][1])
                              );
    }
    return res;
}

// gsl_complex dt_dz(double z, void *p){
//     struct Parameters *params = (struct Parameters *)p;
//     int n = sizeof(params->angles) / sizeof(params->angles[0]);
//     gsl_complex res = 1/params->A;
//     for (int i = 0; i < n; i++){
//         res = res * gsl_complex_pow(1 - t/params->prevertices[i], -params->angles[i]);
//     }
//     return res;
// }

// gsl_complex z_t(gsl_complex a, gsl_complex b, void *p){
//     gsl_function F;
//     struct Parameters *params = (struct Parameters *)p;
//     F.function = &dz_dt;
//     F.params = &params;
//     vector<double> pts;
//     pts.push_back(a);
//     int n = sizeof(params->angles) / sizeof(params->angles[0]);
//     for (int i = 0; i < n; i++){
//         if (a < params->prevertices[i] < b){
//             pts.push_back(params->prevertices[i]);
//         }
//     }
//     pts.push_back(b);
//     size_t npts = std::vector::size(pts);
// }

int main(int argc, char** argv){
    return 0;
}
