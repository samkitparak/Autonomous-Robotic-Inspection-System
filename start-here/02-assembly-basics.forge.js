// A tiny multi-part scene.
// ForgeCAD projects are just code, so assemblies can be built from ordinary
// variables and functions before you move into richer connector workflows.

const spread = Param.number("Spread", 0, { min: 0, max: 40, unit: "mm" });
const postHeight = Param.number("Post Height", 34, { min: 20, max: 70, unit: "mm" });

const base = box(110, 42, 8)
  .color("#4f6f8f");

function makePost(x, color) {
  return cylinder(postHeight, 7)
    .translate(x, 0, 8)
    .color(color);
}

const leftPost = makePost(-28 - spread, "#c76d39");
const rightPost = makePost(28 + spread, "#d9a441");
const cap = box(82 + spread * 2, 18, 8)
  .translate(0, 0, 8 + postHeight)
  .color("#8e5ea2");

return assembly("Assembly Basics")
  .addPart("Base Rail", base)
  .addPart("Left Post", leftPost)
  .addPart("Right Post", rightPost)
  .addPart("Top Cap", cap);
