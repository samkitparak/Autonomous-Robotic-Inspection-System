// Welcome to ForgeCAD.
// Try changing the Width or Height sliders, then watch the model update.
//
// When you are ready to work locally:
//   npm install -g forgecad
//   forgecad login
//   # or, for GitHub/Google accounts: forgecad login --token
//   forgecad project clone start-here
//   cd start-here
//   forgecad studio .
//
// More orientation: https://forgecad.io/docs/welcome

const width = Param.number("Width", 80, { min: 40, max: 140, unit: "mm" });
const depth = Param.number("Depth", 50, { min: 30, max: 90, unit: "mm" });
const height = Param.number("Height", 18, { min: 8, max: 40, unit: "mm" });
const holeRadius = Param.number("Hole Radius", 6, { min: 2, max: 12, unit: "mm" });

const plate = box(width, depth, height)
  .subtract(cylinder(height * 2, holeRadius))
  .color("#5b9bd5");

return { "starter plate": plate };
