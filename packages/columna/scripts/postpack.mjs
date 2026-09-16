import { existsSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const pkgPath = join(root, "package.json");
const bakPath = join(root, "package.json.bak");

if (!existsSync(bakPath)) {
  console.warn("postpack: package.json.bak not found, skipping restore");
  process.exit(0);
}

writeFileSync(pkgPath, readFileSync(bakPath, "utf8"), "utf8");
unlinkSync(bakPath);
