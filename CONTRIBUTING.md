# Contributing to columna

Thanks for your interest in contributing.

## Prerequisites

- **Node.js** 18 or newer (see `engines` in the root `package.json`)
- **pnpm** 9 (this repo pins `packageManager`: `pnpm@9.15.0`)

## Setup

```bash
git clone <repository-url>
cd <repo>
pnpm install
```

## Common commands

| Command | Purpose |
|---|---|
| `pnpm build` | Build all workspace packages |
| `pnpm test` | Run the Vitest suite |
| `pnpm test:stress` | Adversarial load / identity tests (set `STRESS_HEAVY=1` for ~10× rows) |
| `pnpm lint` | ESLint over `packages/` |
| `pnpm typecheck` | TypeScript `--noEmit` in packages that define it |
| `pnpm bench:compare:js` | Refresh `docs/comparison-js.md` (2M-row cross-library suite) |

Filter a single package when useful, for example:

```bash
pnpm --filter @columna/core test
pnpm --filter columna build
```

## Pull requests

- Keep changes focused — avoid unrelated refactors in the same PR.
- Explain **why** the change is needed (bug, API gap, docs, CI).
- Prefer adding or updating tests for bug fixes and behavioral changes.
- For new features: describe the use case, document the public API in the README or `docs/` when relevant, and note any follow-up work.
- Run `pnpm lint`, `pnpm typecheck`, `pnpm test`, and `pnpm build` before opening the PR.
- Do not commit secrets, local `.env` files, or machine-specific paths.
- Mark breaking changes clearly in the PR description.

## Reporting issues

Use the GitHub issue templates. For security vulnerabilities, follow [SECURITY.md](SECURITY.md) instead of opening a public issue.

## Public API changes

`docs/api-surface.json` records every public export and class member; `packages/columna/tests/api-surface.test.ts`
fails when one disappears. Adding is free (run `pnpm api:snapshot` after `pnpm build` to record it); removing or
renaming needs a deprecation cycle first — see [docs/compatibility.md](docs/compatibility.md) — and the snapshot
edit goes in the same commit as the CHANGELOG *Breaking* line.

## Releasing

Only `columna` is published; the `@columna/*` packages are bundled into it. A release is a tag:

```bash
pnpm release:prepare 0.3.0
```

bumps `package.json` / `packages/columna/package.json`, moves the `[Unreleased]` CHANGELOG entries under
`## [0.3.0] - <date>` and checks the section is not empty. Then

```bash
git commit -am "Release 0.3.0" && git tag v0.3.0 && git push --follow-tags
```

[`release.yml`](.github/workflows/release.yml) refuses a tag whose version differs from `package.json` or has no
CHANGELOG section, runs lint / build / typecheck / tests / the clean-consumer install, packs the tarball, creates the
**GitHub Release** with the CHANGELOG section and the tarball's SHA-256, and — when the `NPM_TOKEN` repository
secret exists — publishes to npm with provenance (`--provenance`, so the package page links back to the workflow
run and commit). Pin consumers to the tag or the tarball hash, not to `main`.
