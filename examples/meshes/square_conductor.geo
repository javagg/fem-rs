// Square conductor cross-section benchmark geometry
// Usage: gmsh square_conductor.geo -2 -o square_conductor.msh -format msh4

SetFactory("OpenCASCADE");

// Outer domain: [0,1]²
Rectangle(1) = {0, 0, 0, 1, 1};

// Inner conductor: [0.3, 0.7]²
Rectangle(2) = {0.3, 0.3, 0, 0.4, 0.4};

// Air region = outer minus inner
BooleanDifference{ Surface{1}; Delete; }{ Surface{2}; }

// We now have:
//  Surface 1 = air region (hole where conductor was)
//  The conductor square is Surface 2 (or recreate)

// Simpler approach: just mark sub-regions by physical groups
// and let the FEM code use J_z(x,y) analytically — no boolean needed.
// Just create the full domain and let source be specified in code.
Rectangle(10) = {0, 0, 0, 1, 1};

// Physical groups
Physical Curve("boundary", 1) = {Boundary{Surface{10}:}};  // outer: A_z = 0
Physical Surface("domain", 100) = {10};

Mesh.CharacteristicLengthMax = 0.03;
Mesh.Algorithm = 6;  // Frontal-Delaunay
