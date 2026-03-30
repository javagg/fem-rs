// Coaxial cable cross-section mesh for em_electrostatics example
// Usage: gmsh coaxial.geo -2 -o coaxial.msh -format msh4

SetFactory("OpenCASCADE");

// Geometry parameters
r_inner = 0.2;   // inner conductor radius [m]
r_outer = 0.8;   // outer conductor radius [m]
lc_fine = 0.02;  // mesh size near inner conductor
lc_coarse = 0.05; // mesh size at outer boundary

// Inner circle (conductor surface)
Circle(1) = {0, 0, 0, r_inner, 0, 2*Pi};
Curve Loop(1) = {1};

// Outer circle (outer boundary)
Circle(2) = {0, 0, 0, r_outer, 0, 2*Pi};
Curve Loop(2) = {2};

// Dielectric region between conductors
Plane Surface(1) = {2, 1};

// Physical groups (used by fem-rs for BC assignment)
Physical Curve("inner_conductor", 1) = {1};  // Dirichlet: phi = V_inner
Physical Curve("outer_conductor", 2) = {2};  // Dirichlet: phi = 0
Physical Surface("dielectric", 100) = {1};

// Mesh refinement
Field[1] = Distance;
Field[1].CurvesList = {1};
Field[1].Sampling = 100;
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = lc_fine;
Field[2].SizeMax = lc_coarse;
Field[2].DistMin = 0.01;
Field[2].DistMax = 0.3;
Background Field = 2;
