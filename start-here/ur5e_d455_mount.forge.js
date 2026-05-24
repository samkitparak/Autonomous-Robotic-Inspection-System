// ====================================================================
// UR5e → Intel RealSense D455  |  Minimal Side-Bolt Camera Mount
// ====================================================================
// Flange plate bolts to UR5e end effector.
// Tab hangs below. ¼"-20 bolt goes through the side of the tab.
// Camera threads onto it from the other side, face pointing straight down.
// Bolt head is the stop — no shelf, no cage, nothing else needed.
//
// New TF after printing:
//   --x 0  --y 0  --z -<(FLANGE_THICK + TAB_H) / 1000>
//   --roll 0  --pitch 0  --yaw 0
// ====================================================================

// ── Flange (from Universaladapter.step) ─────────────────────────────
const FLANGE_OD    = param("Flange plate OD (mm)", 82,    { unit: "mm" });
const FLANGE_THICK = param("Flange thickness (mm)", 8,    { min: 5, max: 14, unit: "mm" });
const BOLT_R       = param("Bolt circle radius (mm)", 25.0,  { unit: "mm" });
const BOLT_DIA     = param("M6 clearance dia (mm)", 6.5,  { unit: "mm" });
const PILOT_DIA    = param("Pilot boss dia (mm)", 31.5,   { unit: "mm" });
const PILOT_H      = param("Pilot boss height (mm)", 2.5, { unit: "mm" });

// ── Tab ─────────────────────────────────────────────────────────────
const TAB_W        = param("Tab width (mm)", 30,  { min: 15, max: 80, unit: "mm" });
const TAB_D        = param("Tab depth (mm)", 12,  { min: 8,  max: 25, unit: "mm" });
const TAB_H        = param("Tab height (mm)", 25, { min: 10, max: 50, unit: "mm" });

// ── ¼"-20 side bolt hole ─────────────────────────────────────────────
const QTR_DIA      = param("1/4-20 hole dia (mm)", 6.6, { unit: "mm" });

// ====================================================================
// Build  (Z = 0 at flange bottom face; tab hangs into negative Z)
// ====================================================================

// 1. Flange plate
const flangePlate = cylinder(FLANGE_THICK, FLANGE_OD / 2);

// 2. Pilot boss on top
const pilotBoss = cylinder(PILOT_H, PILOT_DIA / 2)
  .translate(0, 0, FLANGE_THICK);

// 3. Tab hanging below (centred on XY)
const tab = box(TAB_W, TAB_D, TAB_H)
  .translate(0, 0, -TAB_H);

// 4. M6 bolt holes through flange plate
const boltCutter = cylinder(FLANGE_THICK + 0.2, BOLT_DIA / 2)
  .translate(0, 0, -0.1);
const boltHoles = circularLayout(4, BOLT_R, { startDeg: 45 })
  .map(({ x, y }) => boltCutter.translate(x, y, 0));

// 5. ¼"-20 side hole — horizontal, through the full depth of the tab (Y direction)
//    Centred in X and vertically at mid-height of the tab
const sideHole = cylinder(TAB_D + 0.2, QTR_DIA / 2)
  .translate(0, 0, -(TAB_D + 0.2) / 2)  // centre the cylinder at origin along its axis
  .rotateX(90)                            // rotate from Z-axis to Y-axis
  .translate(0, 0, -TAB_H / 2);          // position at mid-height of tab

// ====================================================================
// Combine
// ====================================================================
let part = union(flangePlate, pilotBoss, tab);
part = difference(part, ...boltHoles, sideHole);

return [{ name: "Camera Mount", shape: part.color("#1a7fc1") }];
