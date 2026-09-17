# Columna Studio

Spyder-like web IDE for the `columna` DataFrame library.

## Run

From monorepo root:

```bash
pnpm studio
```

Open in **Chrome/Edge** for Open Folder (File System Access API).

## Features

- Monaco TypeScript editor with `// %%` / `# %%` cell mode
- **Project tree**: Open folder, cwd, relative paths, save to disk handles
- **Debugger**: gutter breakpoints, Debug Run (F9), step over/into/out, call stack, watch
- **Consoles**: multiple kernels, `%who` `%whos` `%time` `%timeit` `%clear` `%hist` `%pwd` `%cd`, Tab completion, rich DataFrame display
- **Find in files** + **F2 rename** across project/session
- **Analysis**: TypeScript diagnostics + Studio heuristics
- Variable Explorer + detached editable DataFrame window
- Plots, History, Help, layout persistence

## Trust model — read before opening someone else's project

Studio is a **REPL**, not a sandbox. Code you run (console input, `F5` on a file, cells, debug runs) executes as
`AsyncFunction` in the page itself, with everything the page can do: read and write the folder you opened via
the File System Access API, use `localStorage` / IndexedDB of the origin, make network requests as the page,
and reach every DataFrame in the session. This is the same power the browser devtools console has.

Consequences:

- Treat a project, notebook or file from someone else exactly like a script you would run on your machine.
  Do not open it in Studio unless you would run it.
- Do not host Studio for other people. The dev server binds to `127.0.0.1` on purpose; `vite --host` turns every
  device on the network into a local user of your browser session.
- Moving execution into a Web Worker or an iframe would **not** create a trust boundary: same origin, same
  storage, same handles passed over `postMessage`, same `fetch`. A real isolation layer (separate origin with
  a restrictive CSP, capability-scoped bridge, no direct handle access, resource limits) would be a separate
  design and is not on the roadmap.

## Shortcuts

| Shortcut | Action |
|---|---|
| Ctrl+Enter | Run selection or current cell |
| Shift+Enter | Run current cell and advance |
| F5 | Run file |
| F9 | Debug run |
| F2 | Rename symbol |
| Click gutter | Toggle breakpoint |
| Console Tab | Complete from scope |
| Console ↑/↓ | History |
| Console Ctrl+Enter | Submit multiline |
