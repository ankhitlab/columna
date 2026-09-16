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
