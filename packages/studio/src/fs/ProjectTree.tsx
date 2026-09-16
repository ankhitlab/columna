import type { FsNode } from './ProjectFS'

export interface ProjectTreeProps {
  tree: FsNode | null
  rootName: string
  cwd: string
  activePath: string | null
  onOpenFolder: () => void
  onCloseFolder: () => void
  onRefresh: () => void
  onOpenFile: (path: string) => void
  onSetCwd: (path: string) => void
}

export function ProjectTree({
  tree,
  rootName,
  cwd,
  activePath,
  onOpenFolder,
  onCloseFolder,
  onRefresh,
  onOpenFile,
  onSetCwd,
}: ProjectTreeProps) {
  return (
    <div className="panel" style={{ border: 'none', height: '100%' }}>
      <div className="panel-header">
        <span>Project{rootName ? ` · ${rootName}` : ''}</span>
        <div className="panel-tools">
          {!tree ? (
            <button className="btn btn-primary" type="button" onClick={onOpenFolder}>
              Open folder
            </button>
          ) : (
            <>
              <button className="btn" type="button" title="Refresh" onClick={onRefresh}>
                ↻
              </button>
              <button className="btn" type="button" onClick={onCloseFolder}>
                Close
              </button>
            </>
          )}
        </div>
      </div>
      {tree && (
        <div className="df-meta" style={{ borderBottom: '1px solid var(--border)' }}>
          <span className="chip" title="Working directory">
            cwd: {cwd === '.' ? rootName || '.' : cwd}
          </span>
        </div>
      )}
      <div className="panel-body">
        {!tree ? (
          <p className="muted" style={{ padding: 12 }}>
            Open a folder (Chrome/Edge) to browse the project tree. Paths are relative to the folder
            root.
          </p>
        ) : (
          <ul className="file-list project-tree">
            <TreeNode
              node={tree}
              depth={0}
              activePath={activePath}
              cwd={cwd}
              onOpenFile={onOpenFile}
              onSetCwd={onSetCwd}
            />
          </ul>
        )}
      </div>
    </div>
  )
}

function TreeNode({
  node,
  depth,
  activePath,
  cwd,
  onOpenFile,
  onSetCwd,
}: {
  node: FsNode
  depth: number
  activePath: string | null
  cwd: string
  onOpenFile: (path: string) => void
  onSetCwd: (path: string) => void
}) {
  if (node.kind === 'file') {
    return (
      <li
        className={`file-item ${activePath === node.path ? 'active' : ''}`}
        style={{ paddingLeft: 10 + depth * 12 }}
        onClick={() => onOpenFile(node.path)}
      >
        <span className="file-name">{node.name}</span>
      </li>
    )
  }

  return (
    <>
      {node.path !== '.' && (
        <li
          className={`file-item ${cwd === node.path ? 'active' : ''}`}
          style={{ paddingLeft: 10 + depth * 12 }}
          onClick={() => onSetCwd(node.path)}
          onDoubleClick={() => onSetCwd(node.path)}
          title="Set as cwd"
        >
          <span className="file-name">📁 {node.name}/</span>
        </li>
      )}
      {(node.children ?? []).map((child) => (
        <TreeNode
          key={child.path}
          node={child}
          depth={node.path === '.' ? depth : depth + 1}
          activePath={activePath}
          cwd={cwd}
          onOpenFile={onOpenFile}
          onSetCwd={onSetCwd}
        />
      ))}
    </>
  )
}
