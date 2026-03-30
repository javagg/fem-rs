// Transformer cross-section for em_magnetostatics_2d --case transformer
// Usage: gmsh transformer.geo -2 -o transformer.msh -format msh4

SetFactory("OpenCASCADE");

// Outer domain
Rectangle(1) = {0, 0, 0, 1, 1};

// Iron core (outer frame)
Rectangle(2) = {0.05, 0.05, 0, 0.90, 0.90};

// Window opening in core
Rectangle(3) = {0.25, 0.15, 0, 0.50, 0.70};

// Core = outer frame minus window
BooleanDifference(4) = { Surface{2}; Delete; }{ Surface{3}; Delete; };

// Primary winding
Rectangle(5) = {0.28, 0.18, 0, 0.16, 0.64};

// Secondary winding
Rectangle(6) = {0.56, 0.18, 0, 0.16, 0.64};

// Air = domain minus core minus windings
BooleanDifference(10) = {
    Surface{1}; Delete;
}{ Surface{4, 5, 6}; };

// Physical groups
Physical Curve("outer_boundary",   1) = {Boundary{Surface{10}:}};
Physical Surface("air",          100) = {10};
Physical Surface("iron_core",    200) = {4};
Physical Surface("winding_pos",  300) = {5};   // +J_z
Physical Surface("winding_neg",  400) = {6};   // -J_z

Mesh.CharacteristicLengthMax = 0.025;
Mesh.Algorithm = 6;
