#include <cmath>
#include <iostream>
#include <string>

int factorial(int n){
    if (n <= 1) return 1;

    return n * factorial(n - 1);
}

double recurrence(BernsteinPolynomial b, int t){
    int i = b.index();
    int p = b.degree();
    return (1 - t) * BernsteinPolynomial(i, p - 1) + t * BernsteinPolynomial(i - 1, p - 1)
}

double elevate(BernsteinPolynomial b, int t){
    int i = b.index();
    int p = b.degree();
    return (1 + i) / (1 + p) * BernsteinPolynomial(i + 1, p + 1) + (1 + p - i) / (1 + p) * BernsteinPolynomial(i, p + 1);
}

double derivative(BernsteinPolynomial b, int m, float t=0, tol=1e-8){
    int i = b.index();
    int p = b.degree();
    if (m == 1) {
        if (std::abs(t - i/p) < tol) return 0; // unimodal
        return p * (BernsteinPolynomial(i - 1, p - 1) - BernsteinPolynomial(i, p - 1));
    }
    if ((m <= i - 1) && (t < tol)) return 0;
    if ((m <= p - i - 1) && (1 - t < tol)) return 0;

    return p * ( derivative(BernsteinPolynomial(i - 1, p - 1), m - 1) - derivative(BernsteinPolynomial(i, p - 1), m - 1))
}

class BernsteinPolynomial {
public:
    BernsteinPolynomial(int index, int degree){ i = index; p = degree; }
    ~BernsteinPolynomial(){};
    eval(int t){ return 1.0 * factorial(p) / factorial(i) * std::Pow(t, i) * std::Pow(1 - t, p - i)}
    int index() { return i; };
    int degree() { return p; };
private:
    int i, p;
};

class Knot {
public:
    Knot(){};
    ~Knot(){};
private:

};
class BasisSpline {
public:
    BasisSpline(std::string name, int order){ type = name; p = order; };
    ~Basis(){};
    int order(){ return p; };
    std::string name(){ return type; };
private:
    int p;
    std::string  type;
};