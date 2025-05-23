//Merge "output/segmentation/cam-repaired.stl";
Coherence;
//CreateTopology;
//ClassifySurfaces{Pi, 1, 1, Pi};
//CreateGeometry;

//s() = Surface{:}; // Get all the surfaces
//Surface Loop(1) = s(); // Creating a surface loop to be used to generate the volume
//Volume(1) = 1; // Volumetric mesh to generate
//v() = Volume{:};
//Coherence;
Lx = 499;
Ly = 499;
Lz = 201;
Lsep = 100;
Point(1) = {0, 0, 0};
Point(2) = {Lx, 0, 0};
Point(3) = {Lx, Ly + Lsep, 0};
Point(4) = {0, Ly + Lsep, 0};
Point(5) = {0, 0, Lz};
Point(6) = {Lx, 0, Lz};
Point(7) = {Lx, Ly + Lsep, Lz};
Point(8) = {0, Ly + Lsep, Lz};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

Line(9) = {4, 8};
Line(10) = {5, 1};
Line(11) = {2, 6};
Line(12) = {3, 7};
Coherence;
Curve Loop(1) = {1, 2, 3, 4}; Plane Surface(1) = {1};
Curve Loop(2) = {5, 6, 7, 8}; Plane Surface(2) = {2};
Curve Loop(3) = {9, 8, 10, -4}; Plane Surface(3) = {3};
Curve Loop(4) = {10, 1, 11, -5}; Plane Surface(4) = {4};
Curve Loop(5) = {11, 6, -12, -2}; Plane Surface(5) = {5};
Curve Loop(6) = {12, 7, -9, -3}; Plane Surface(6) = {6};
Surface Loop(1) = {1, 2, 3, 4, 5, 6};
Volume (1) = {1};
Physical Volume(1) = {1};
Coherence;
Save "cam.msh";
