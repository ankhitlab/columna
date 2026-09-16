import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const pkgPath = join(root, "package.json");
const bakPath = join(root, "package.json.bak");

const pkg = JSON.parse(readFileSync(pkgPath, "utf8"));
writeFileSync(bakPath, readFileSync(pkgPath, "utf8"), "utf8");

const depFields = ["dependencies", "devDependencies", "optionalDependencies"];

for (const field of depFields) {
  const deps = pkg[field];
  if (!deps || typeof deps !== "object") continue;
  for (const name of Object.keys(deps)) {
    if (name.startsWith("@columna/") || deps[name] === "workspace:*" || String(deps[name]).startsWith("workspace:")) {
      delete deps[name];
    }
  }
  if (Object.keys(deps).length === 0) {
    delete pkg[field];
  }
}

writeFileSync(pkgPath, JSON.stringify(pkg, null, 2) + "\n", "utf8");
