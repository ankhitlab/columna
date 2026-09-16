import type { SessionFile } from '../types'

export interface FileSidebarProps {
  files: SessionFile[]
  activeId: string | null
  onSelect: (id: string) => void
  onCreate: () => void
  onRemove: (id: string) => void
}

export function FileSidebar({ files, activeId, onSelect, onCreate, onRemove }: FileSidebarProps) {
  return (
    <div className="panel">
      <div className="panel-header">
        <span>Files</span>
        <button className="btn" onClick={onCreate}>
          New
        </button>
      </div>
      <div className="panel-body">
        <ul className="file-list">
          {files.map((f) => (
            <li
              key={f.id}
              className={`file-item ${activeId === f.id ? 'active' : ''}`}
              onClick={() => onSelect(f.id)}
            >
              <span className="file-name">{f.name}</span>
              {files.length > 1 && (
                <button
                  className="btn btn-ghost"
                  onClick={(e) => {
                    e.stopPropagation()
                    onRemove(f.id)
                  }}
                >
                  ×
                </button>
              )}
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
