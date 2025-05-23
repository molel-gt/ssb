Merge "output/segmentation/cam-repaired.stl";
Coherence;
//CreateTopology;
//ClassifySurfaces{Pi, 1, 1, Pi};
//CreateGeometry;
s() = Surface{:}; // Get all the surfaces
Surface Loop(1) = s(); // Creating a surface loop to be used to generate the volume
Volume(1) = 1; // Volumetric mesh to generate
v() = Volume{:};

Coherence;
Save "cam.msh";
