#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#define TOL 1e-12

int factorial(int n){
    if (n <= 1) return 1;

    return n * factorial(n - 1);
}

class B {
public:
    B(int index, int degree){ i = index; p = degree; }
    ~B(){};
    eval(int t){ return 1.0 * factorial(p) / factorial(i) * std::Pow(t, i) * std::Pow(1 - t, p - i)}
    int index() { return i; };
    int degree() { return p; };
private:
    int i, p;
};

double recurrence(B b, int t){
    int i = b.index();
    int p = b.degree();

    return (1 - t) * B(i, p - 1) + t * B(i - 1, p - 1)
}

double elevate(BernsteinPolynomial b, int t){
    int i = b.index();
    int p = b.degree();

    return (1 + i) / (1 + p) * B(i + 1, p + 1) + (1 + p - i) / (1 + p) * B(i, p + 1);
}

double derivative(BernsteinPolynomial b, int m, float t=0){
    int i = b.index();
    int p = b.degree();

    if (m == 1) {
        if (std::abs(t - i/p) < TOL) return 0; // unimodal
        return p * (B(i - 1, p - 1) - B(i, p - 1));
    }
    if ((m <= i - 1) && (t < tol)) return 0;
    if ((m <= p - i - 1) && (1 - t < TOL)) return 0;

    return p * ( derivative(B(i - 1, p - 1), m - 1) - derivative(B(i, p - 1), m - 1))
}

class BSpline {
    /*
    index: i
    degree: p
    knot sequence: Ξ
    */
public:
    BSpline(int index, int degree, std::vector<int>& knots){
        i = index;
        p = degree;
        std::copy(knots.begin(), knots.end(), Ξ);
    };
    ~BSpline(){};
    int degree(){ return p; };
    int index(){ return i; };
    std::vector<int> knots(){ return Ξ; }
    double eval(float t){
        if (p == 0){
            if (Ξ[i] <= t < Ξ[i + 1]) return 1;
            return 0;
        }
        else {
            return ω(i, t) * BSpline(i, p - 1, Ξ) + (1 - ω(i + 1, t)) * BSpline(i + 1, p - 1, Ξ)
        }
    };
private:
    int p;
    int i;
    std::vector<int> Ξ;
    double ω(int x, double t){
        double top, bottom;
        top = t - Ξ[x];
        bottom = Ξ[x + p] - Ξ[x];
        if (std::abs(std::abs(bottom) - TOL) <= 0) return 0;
        return top / bottom;
    }
};