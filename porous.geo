
//+ file_name inserted above
SetFactory("OpenCASCADE");
Mesh.CharacteristicLengthMax = 0.1;
Physical Volume(1) = {1};
Merge Str(file_name);
Coherence;