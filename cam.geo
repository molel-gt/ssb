Merge "output/segmentation/cam/201-201-201/0-0-0/3_surf_smooth_mesh.ply";
//Nerge "cam.msh";
//Coherence;
CreateTopology;
ClassifySurfaces{Pi, 1, 1, Pi};
CreateGeometry;

s() = Surface{:}; // Get all the surfaces
Surface Loop(1) = s(); // Creating a surface loop to be used to generate the volume
Coherence;
Volume(1) = {1}; // Volumetric mesh to generate
Physical Volume(1) = {1};
//v() = Volume{:};

Coherence;
Save "cam.msh";
