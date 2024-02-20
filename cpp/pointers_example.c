#include <stdio.h>

void Swap(int *firstVal, int *secondVal);

int main(){
    int valueA = 3;
    int valueB = 4;

    printf("Before swap ");
    printf("valueA = %d and valueB = %d\n", valueA, valueB);
    
    Swap(&valueA, &valueB);

    printf("After swap ");
    printf("valueA = %d and valueB = %d\n", valueA, valueB);
}

void Swap(int *firstVal, int *secondVal){
    int tempVal;

    tempVal = *firstVal;
    *firstVal = *secondVal;
    *secondVal = tempVal;
}