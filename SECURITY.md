# Security Policy

## Supported versions

This project has not yet published a stable release line. Until then, security fixes target the default branch (`main` / `master`) and the latest published `columna` version on npm (currently `0.2.0`).

After the first public release, this section should list which released versions still receive security updates.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Preferred options:

1. Enable and use **GitHub Private Vulnerability Reporting** on this repository (Settings → Code security and analysis → Private vulnerability reporting), then submit a private advisory.
2. If Private Vulnerability Reporting is not available, contact the repository maintainers through a private channel they publish (for example a security contact on the GitHub org/profile). Do not invent or use unofficial addresses.

Include as much detail as you can: affected package(s), version/commit, reproduction steps, and impact.

We aim to acknowledge reports promptly and will coordinate a fix and disclosure timeline with you.

## Threat model notes

- **Untrusted data files.** CSV / JSON / Parquet readers are pure JavaScript with bounds-checked typed arrays; malformed input raises an error or yields `null` cells, never out-of-bounds memory access. Column names are treated as data (every reader — CSV, JSON, Excel, Parquet, SQL, Kafka — builds rows through one primitive, `setRowField`): a column called `__proto__` or `constructor` becomes an ordinary property and cannot alter prototypes.
- **Paths and URLs from outside the program.** Readers accept a bare string and guess whether it is content, a path or a URL; that guess is convenience, not a boundary. Server-side code must tag the source (`{ text }` / `{ path }` / `{ url }` or `mode`) and set a policy — `allowedHosts`, `denyPrivateHosts`, `allowedDirs`, `maxBytes`, `timeoutMs`, `signal` — per call or once via `setIoPolicy()` (a floor that per-call options cannot widen). Without a policy the library reads whatever the process can reach; that is the documented default, not a vulnerability. `denyPrivateHosts` blocks loopback/private/link-local **IP literals** (including IPv4-mapped IPv6 such as `[::ffff:127.0.0.1]`) and common private hostnames; it does **not** resolve DNS — pass a pinning `fetch` if rebinding is in scope.
- **Excel input.** `readExcel` / `fromExcel` delegate to SheetJS, pinned to the patched build `xlsx@0.20.3` from cdn.sheetjs.com (the npm registry line stopped at 0.18.5, which carries CVE-2023-30533 prototype pollution and CVE-2024-22363 ReDoS). Keep the tarball URL in `packages/core/package.json` when bumping; do not downgrade to the npm `^0.18` line.
- **Expression engine.** Fused arithmetic kernels are compiled with `new Function` from a fixed grammar: column references become array indices and only numeric literals reach the generated source; string values never do. A CSP without `unsafe-eval` falls back to closures automatically.
- **Formulas and designs.** `parseFormula` caps crossed factors (`a*b*…`) at 12 so a hostile formula cannot expand to 2^k terms; DOE constructors validate their inputs before building matrices.
- **SQL / Kafka.** Query parameters are always bound through the driver; the library never interpolates values into SQL. Driver packages are loaded by fixed name, never from user input.
- **CSV export.** `toCsv()` / `writeCsv()` write cell text unchanged (like pandas / polars). A text cell starting with `=` `+` `-` `@` (or tab / CR) is executed as a formula by Excel-like applications that open the file — a risk in the consumer, not in the serializer. For exports of untrusted text use `{ escapeFormulas: true }`: such cells (and header names) get a `'` prefix and are quoted.
- **Spill directory.** When `MemoryPolicy.spill` is enabled (Node), intermediate tables are written under `spillDir` (default `os.tmpdir()/columna-spill`). Treat that directory like any other filesystem sink: confine it with the same care as `IoPolicy.allowedDirs` for reads (set an application-owned `spillDir` inside an allowed tree; do not point it at untrusted shared locations). Spill files are unlinked after the operator finishes; a crash may leave orphans in `spillDir`.
