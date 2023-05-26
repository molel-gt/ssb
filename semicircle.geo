SetFactory('OpenCASCADE');
Mesh.CharacteristicLengthMin = 0.0001;
Mesh.CharacteristicLengthMax = 0.1;
// Mesh.SubdivisionAlgorithm  = 1;
//Mesh.Hexahedra = 1;

Point(1) = {5, 0, 0};
Point(2) = {5, 5, 0};
Point(3) = {5, 10, 0};
Point(4) = {0, 5, 0};


Circle(1) = { 1, 2, 4};
Circle(2) = { 4, 2, 3};

Line(3) = {1, 3};
//Line(2) = {2, 3};
//Line(3) = {3, 4};
//Line(4) = {4, 1};

Physical Line(1) = {1, 2};
Physical Line(2) = {3};
//Physical Line(3) = {1, 3};
//Curve Loop(1) = {1:4};
Line Loop(1) = {1, 2, 3};
Plane Surface(1) = {1};
Physical Surface(1) = {1};

//Rotate{{0, 0, 1}, {0,2,5}, -Pi/3}{ Surface{2}; }