#include <stdio.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
// #include <gsl/gsl_sf_hyperg.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include <iomanip>
#include <stdexcept>

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
    return Kkprime / Kk - 2 * h / L;
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
    int max_iters = 1000;
    double eps = std::numeric_limits<double>::epsilon();
    double max_k = 1.0 - 2.5 * eps;
    double min_k = 0.0;
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

        // error if new k is greater than 1.0
        if (knew < min_k || knew > max_k) {
            knew = k;
            gamma *= 0.5;
        }

        _kprime = get_kprime(knew);
        _Kk = M_PI_2 * gsl_sf_hyperg_2F1(a, b, c, pow(knew, 2));
        _Kkprime = M_PI_2 * gsl_sf_hyperg_2F1(a, b, c, pow(_kprime, 2));
        error = fabs(2.0 * h / L - _Kkprime/_Kk) / (2.0*h/L);
        k = knew;
        iters ++;
    }
    printf("Error is: %lf after %d iterations\n", error, iters);
    if (error > tol) { throw std::invalid_argument("Could not converge\n"); } else { return k; }
}

double get_a(double w_over_L, double k){
    mode_t prec_mode = GSL_PREC_DOUBLE;
    double phi = 0.5 * M_PI_2; // pi/4
    double phi_new;
    double tol = 1e-3;
    double error = tol + 1.0;
    int max_iters = 1000;
    double eps = std::numeric_limits<double>::epsilon();
    double iters = 0;
    double gamma = 0.25;
    double ratio = 1 - 2 * w_over_L;
    double Kk = gsl_sf_ellint_Kcomp(k, prec_mode);
    while (error > tol && iters < max_iters){
        phi_new = phi - gamma * Kk * sqrt(1 - pow(k, 2)*pow(sin(phi), 2)) *(-ratio + (1.0/Kk) * gsl_sf_ellint_F(phi, k, prec_mode));
        // reset phi when exceeds bounds
        if (fabs(phi_new) > M_PI_2) {phi_new = phi - eps; gamma *= 0.5;}
        error = fabs(gsl_sf_ellint_F(phi_new, k, prec_mode)/Kk - ratio)/ratio;
        phi = phi_new;
        iters++;
    }
    if (error > tol) { throw std::invalid_argument("Could not converge\n"); } else { double a = sin(phi); return a; }
}

int main(int argc, char **argv){
    std::vector<double> w_over_L = {0.49, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01};
    double k, b, h_L, w_L;
    mode_t prec_mode = GSL_PREC_DOUBLE;
    double eps = std::numeric_limits<double>::epsilon();
    fstream fin;
    fin.open("aspect-ratios.txt", ios::in);

    vector<double> aspectratios;
    if (!fin.is_open()) {
        std::cerr << "Error opening the file!" << std::endl;
        return 1;
    }
    string line;
    while (getline(fin, line)) {
        aspectratios.push_back(std::stod(line));
    }

    fin.close();
    ofstream fout, fout2;
    fout.open("geometric-ratios.csv");
    fout2.open("geometric-factors.csv");
    fout2 << "A,L,h,w,k,b,a,h_over_L,w_over_L\n";
    fout  << "aspect_ratio,b\n";
    for (auto & element : aspectratios){
        h_L = element;
        try {
            k = get_k(element, 1.0);
            b = 1.0/k;
            fout << element << "," << std::setprecision(16) << b << "\n";
        }
        catch (const std::invalid_argument& e) {
            k = 1;
            b = 1 / k;
            fout << element << "," << "-\n";
            std::cout << "Could not converge for h/L = " << element << "\n";
        }

        for (auto & element : w_over_L)
            {
                try
                {
                    if (fabs(k - 1) > eps){
                        double kprime = get_kprime(k);
                        double h = gsl_sf_ellint_Kcomp(kprime, prec_mode);
                        w_L = element;
                        double L = 2 * gsl_sf_ellint_Kcomp(k, prec_mode);
                        double a = get_a(w_L, k);
                        double w = 0.5 * L - gsl_sf_ellint_F(asin(a), k, prec_mode);
                        printf("w/L: %lf, a: %lf, b: %lf\n", w_L, a, b);
                        fout2 << std::setprecision(16) << 1/L << "," << std::setprecision(16) << L << ","<< std::setprecision(16) << h << "," << std::setprecision(16) << w << "," << std::setprecision(16) << k << "," << std::setprecision(16) << b << "," << std::setprecision(16) << a << "," << std::setprecision(16) << h_L << "," << std::setprecision(16) << w_L << "\n";
                }
            }
            catch (const std::invalid_argument& e) {
                std::cout << "Could not converge for w/L = " << w_L << " and for k = " << k << "\n";
            } 
        }
    }
    fout.close();
    fout2.close();

    return 0;
}
