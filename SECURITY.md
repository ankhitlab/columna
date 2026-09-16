# Security Policy

## Supported versions

This project has not yet published a stable release line. Until then, security fixes target the default branch (`main` / `master`) at version `0.1.0` in the monorepo.

After the first public release, this section should list which released versions still receive security updates.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Preferred options:

1. Enable and use **GitHub Private Vulnerability Reporting** on this repository (Settings → Code security and analysis → Private vulnerability reporting), then submit a private advisory.
2. If Private Vulnerability Reporting is not available, contact the repository maintainers through a private channel they publish (for example a security contact on the GitHub org/profile). Do not invent or use unofficial addresses.

Include as much detail as you can: affected package(s), version/commit, reproduction steps, and impact.

We aim to acknowledge reports promptly and will coordinate a fix and disclosure timeline with you.

## Threat model notes

- **Untrusted data files.** CSV / JSON / Parquet readers are pure JavaScript with bounds-checked typed arrays; malformed input raises an error or yields `null` cells, never out-of-bounds memory access. Column names are treated as data: a column called `__proto__` or `constructor` becomes an ordinary property and cannot alter prototypes.
- **Excel input.** `readExcel` / `fromExcel` delegate to SheetJS, pinned to the patched build `xlsx@0.20.3` from cdn.sheetjs.com (the npm registry line stopped at 0.18.5, which carries CVE-2023-30533 prototype pollution and CVE-2024-22363 ReDoS). Keep the tarball URL in `packages/core/package.json` when bumping; do not downgrade to the npm `^0.18` line.
- **Expression engine.** Fused arithmetic kernels are compiled with `new Function` from a fixed grammar: column references become array indices and only numeric literals reach the generated source; string values never do. A CSP without `unsafe-eval` falls back to closures automatically.
- **Formulas and designs.** `parseFormula` caps crossed factors (`a*b*…`) at 12 so a hostile formula cannot expand to 2^k terms; DOE constructors validate their inputs before building matrices.
- **SQL / Kafka.** Query parameters are always bound through the driver; the library never interpolates values into SQL. Driver packages are loaded by fixed name, never from user input.
- **Studio.** The REPL executes user-supplied JavaScript by design; do not expose a Studio instance to untrusted users.
