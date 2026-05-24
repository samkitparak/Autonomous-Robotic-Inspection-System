// A small parametric bracket.
// This file shows the normal ForgeCAD rhythm:
// parameters -> primitives -> booleans -> named result.

const width = Param.number("Width", 90, { min: 60, max: 150, unit: "mm" });
const depth = Param.number("Depth", 64, { min: 42, max: 100, unit: "mm" });
const height = Param.number("Height", 70, { min: 40, max: 120, unit: "mm" });
const plate = Param.number("Plate", 8, { min: 4, max: 14, unit: "mm" });
const holeRadius = Param.number("Hole Radius", 4, { min: 2, max: 8, unit: "mm" });
const spacing = Param.number("Hole Spacing", 48, { min: 24, max: 80, unit: "mm" });

const base = box(width, depth, plate);
const upright = box(width, plate, height).translate(0, -depth / 2 + plate / 2, plate);

const baseHoles = [-spacing / 2, spacing / 2].map((x) =>
  cylinder(plate * 3, holeRadius).translate(x, depth * 0.18, -plate)
);

const uprightHoles = [-spacing / 2, spacing / 2].map((x) =>
  cylinder(plate * 3, holeRadius)
    .pointAlong([0, 1, 0])
    .translate(x, -depth / 2 - plate, plate + height * 0.58)
);

const bracket = union(base, upright)
  .subtract(...baseHoles, ...uprightHoles)
  .color("#6aa84f");

return { "parametric bracket": bracket };
