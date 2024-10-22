#include <complex.h>
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

const int N = 4;  // default number of vertices
using namespace std;

struct Parameters {
    gsl_complex A;
    std::pair<gsl_complex, double> coefs[N];
    // double angles[]; // multiples of pi
    // double prevertices[];
};

gsl_complex dz_dt(gsl_complex t, void *p){
    struct Parameters *params = (struct Parameters *)p;
    // int n = sizeof(params->coefs) / (sizeof(params->coefs[0].first));
    gsl_complex res = params->A;
    for (int i=0; i < N; i++){
        res = gsl_complex_mul(res,
                              gsl_complex_pow_real(
                                              gsl_complex_add_real(gsl_complex_div(gsl_complex_negative(t), params->coefs[i].first), 1),
                                              params->coefs[i].second)
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

gsl_complex z_t(gsl_complex a, gsl_complex b, void *p){
    gsl_complex result;
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
    gsl_function F;
    double error;
    struct Parameters *params = (struct Parameters *)p;
    F.function = &dz_dt;
    F.params = &params;
    vector<gsl_complex> pts;
    pts.push_back(a);
    // int n = sizeof(params->angles) / sizeof(params->angles[0]);
    for (int i = 0; i < N; i++){
        if (a < params->coefs[i].first < b){
            pts.push_back(params->coefs[i].first);
        }
    }
    pts.push_back(b);
    size_t npts = pts.size();
    gsl_integration_qagp(&F, pts, npts, 0, 1e-7, 1000,
                        w, &result, &error);
    return result;
}

int main(int argc, char** argv){
    struct Parameters *params;
    params->A = gsl_complex_rect(1.0, 0);
    gsl_complex t0 = gsl_complex_rect(-1.05, 0);
    gsl_complex t1 = gsl_complex_rect(-1.0, 0);
    gsl_complex t2 = gsl_complex_rect(1.0, 0);
    gsl_complex t3 = gsl_complex_rect(1.05, 0);
    params->coefs[0] = std::pair<gsl_complex, double>(t0, -0.5);
    params->coefs[1] = std::pair<gsl_complex, double>(t1, -0.5);
    params->coefs[2] = std::pair<gsl_complex, double>(t2, -0.5);
    params->coefs[3] = std::pair<gsl_complex, double>(t3, -0.5);
    gsl_complex a = gsl_complex_rect(0.0, 0.0);
    gsl_complex b = gsl_complex_rect(1.0, 0.0);
    gsl_complex integ = z_t(a, b, params);

    return 0;
}
