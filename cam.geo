Merge "output/segmentation/cam.stl";
//SetOrder 1;
//Nerge "cam.msh";
//Mesh.MeshSizeMin = 1;
//Coherence;
//CreateTopology;
ClassifySurfaces{180/180.*Pi, 1, 1, Pi};
CreateGeometry;

s() = Surface{:}; // Get all the surfaces
Surface Loop(1) = s(); // Creating a surface loop to be used to generate the volume
Coherence;
Volume(1) = {1}; // Volumetric mesh to generate
Physical Volume(1) = {1};
//v() = Volume{:};

//Coherence;
Save "cam.msh";
