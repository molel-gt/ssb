#include <stdio.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_hyperg.h>
#include <fstream>
#include <iostream>
using namespace std;


double get_kprime(double k){
    return sqrt(1 - pow(k, 2));
}

double dkprime_dk(double k){
    return -1.0 * pow(1 - pow(k, 2), -0.5);
}

double dKk_dk(double a, double b, double c, double k){
    return M_PI_2 * gsl_sf_hyperg_2F1(a + 1, b + 1, c + 1, pow(k, 2));
}

double get_F(double a, double b, double c, double k, double h, double L){
    double kprime = get_kprime(k);
    double Kkprime = M_PI_2 * gsl_sf_hyperg_2F1(a, b, c, pow(kprime, 2));
    double Kk =  M_PI_2 * gsl_sf_hyperg_2F1(a, b, c, pow(k, 2));
    return  Kkprime / Kk - 2 * h / L;
}

double get_Fprime(double a, double b, double c, double k){
    double kprime = get_kprime(k);
    double Kkprime = M_PI_2 * gsl_sf_hyperg_2F1(a, b, c, pow(kprime, 2));
    double Kk =  M_PI_2 * gsl_sf_hyperg_2F1(a, b, c, pow(k, 2));
    return 1.0/Kk * dKk_dk(a, b, c, kprime) * dkprime_dk(k) - Kkprime * dKk_dk(a, b, c, k) / pow(Kk, 2);
}

double get_k(double h, double L){
    double k = 0.75;
    double tol = 1e-3;
    double error = tol + 1.0;
    int max_iters = 100;
    int iters = 0;
    double knew = k;
    double a = 0.5;
    double b = 0.5;
    double c = 1.0;
    double _kprime;
    double _Kk, _Kkprime;
    double gamma = 0.5;

    while (error > tol && iters < max_iters){
        knew = k - gamma * get_F(a, b, c, k, h, L) / get_Fprime(a, b, c, k);
        knew = fmin(knew, 1.0 - 1e-23);
        _kprime = get_kprime(knew);
        _Kk = M_PI_2 * gsl_sf_hyperg_2F1(a, b, c, pow(knew, 2));
        _Kkprime = M_PI_2 * gsl_sf_hyperg_2F1(a, b, c, pow(_kprime, 2));
        error = fabs(2.0 * h / L - _Kkprime/_Kk) / (2.0*h/L);
        k = knew;
        iters ++;
    }
    printf("Error is: %lf after %d iterations\n", error, iters);
    return k;
}

// double integrate_q(double a, double b, double x){
//     double k = a/b;
//     double var = asin(x/a);
//     return a * sqrt(1 - pow(x, 2)/pow(a, 2)) * sqrt(pow(b, 2) - pow(x, 2)) * gsl_sf_ellint_Ecomp(k)/(sqrt(pow(x, 2) - pow(a, 2))*sqrt(1 - pow(x, 2)/pow(b, 2)));
// }

// double q_avg(double a, double b){
//     return integrate_q(a, b, 1) - integrate_q(a, b, a);
// }

int main(int argc, char **argv){
    double k;
    fstream fin;
    fin.open("aspect-ratios.txt", ios::in);
    fin.close();
    k = get_k(0.1, 1.0);
    // double a = 0.875;
    double b = 1/k;
    // double average_q = q_avg(a, b);
    // printf("Average flow rate: %lf\n", average_q);
    printf("Optimized k is: %lf\n", k);
    return 0;
}
