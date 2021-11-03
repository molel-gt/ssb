
//+ size inserted above
SetFactory("OpenCASCADE");

//+ numbering counterclockwise from bottom/left
Point(1) = {0, 0, 0};
Point(2) = {size, 0, 0};
Point(3) = {size, size, 0};
Point(4) = {0, size, 0};
Point(5) = {0, 0, size};
Point(6) = {size, 0, size};
Point(7) = {size, size, size};
Point(8) = {0, size, size};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};       
Line(4) = {4, 1};
Line(5) = {1, 5};
Line(6) = {2, 6};
Line(7) = {3, 7};
Line(8) = {4, 8};
Line(9) = {5, 6};
Line(10) = {6, 7};
Line(11) = {7, 8};
Line(12) = {8, 5};       

Physical Line(1) = {1};
Physical Line(2) = {2};
Physical Line(3) = {3};
Physical Line(4) = {4};
Physical Line(5) = {5};
Physical Line(6) = {6};
Physical Line(7) = {7};
Physical Line(8) = {8};
Physical Line(9) = {9};
Physical Line(10) = {10};
Physical Line(11) = {11};
Physical Line(12) = {12};

Line Loop(4) = {4, 3, 2, 1};
Line Loop(5) = {9, 10, 11, 12};
Line Loop(6) = {1, 6, -9, -5};
Line Loop(7) = {3, 8, -11, -7};
Line Loop(8) = {4, 5, -12, -8};
Line Loop(9) = {2, 7, -10, -6};

Plane Surface(8) = {4};
Plane Surface(9) = {5};
Plane Surface(10) = {6};
Plane Surface(11) = {7};
Plane Surface(12) = {8};
Plane Surface(13) = {9};

Physical Surface(1) = {8};  // Z0
Physical Surface(2) = {9};  // Z1
Physical Surface(3) = {10};  // Y0
Physical Surface(4) = {11};  // Y1
Physical Surface(5) = {12};  // X0
Physical Surface(6) = {13};  // X1

Physical Volume(1) = {1};
Merge "porous-solid.1.vtk";
Coherence;