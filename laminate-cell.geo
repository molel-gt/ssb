SetFactory("OpenCASCADE");
Rp = 10;

Cylinder(1) = {2 * Rp, 2 * Rp, 0, 0, 0, Rp, 2 * Rp}; 
Torus(2) = {2 * Rp, 2 * Rp, Rp, 2 * Rp, Rp, 2 * Pi};
Cylinder(3) = {2 * Rp, 2 * Rp, 0, 0, 0, 50, Rp};
BooleanUnion(4) = {Volume{1}; Delete; }{Volume{3}; Delete;};
BooleanDifference{Volume{4}; Delete; } {Volume{2}; Delete;}
