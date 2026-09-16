import { uid } from '../types'

export type FsNodeKind = 'file' | 'dir'

export interface FsNode {
  name: string
  path: string
  kind: FsNodeKind
  handle?: FileSystemHandle
  children?: FsNode[]
}

export interface ProjectFileBuffer {
  id: string
  path: string
  name: string
  content: string
  handle?: FileSystemFileHandle
  dirty: boolean
}

type DirPickerWindow = Window & {
  showDirectoryPicker?: (options?: unknown) => Promise<FileSystemDirectoryHandle>
}

const TEXT_EXT = /\.(ts|tsx|js|jsx|mjs|cjs|json|md|txt|csv|css|html|yml|yaml)$/i
const SKIP_DIRS = new Set(['node_modules', '.git', 'dist', 'build', '.next', 'coverage'])

export class ProjectFS {
  root: FileSystemDirectoryHandle | null = null
  rootName = ''
  cwd = '.'
  tree: FsNode | null = null
  buffers: ProjectFileBuffer[] = []
  activeId: string | null = null
  private listeners = new Set<() => void>()

  subscribe(fn: () => void): () => void {
    this.listeners.add(fn)
    return () => this.listeners.delete(fn)
  }

  private emit(): void {
    for (const fn of this.listeners) fn()
  }

  get isOpen(): boolean {
    return !!this.root
  }

  getActive(): ProjectFileBuffer | undefined {
    return this.buffers.find((b) => b.id === this.activeId)
  }

  async openFolder(): Promise<boolean> {
    const w = window as DirPickerWindow
    if (typeof w.showDirectoryPicker !== 'function') {
      throw new Error('Open Folder requires Chrome/Edge File System Access API')
    }
    const handle = await w.showDirectoryPicker({ mode: 'readwrite' })
    this.root = handle
    this.rootName = handle.name
    this.cwd = '.'
    this.buffers = []
    this.activeId = null
    await this.refreshTree()
    this.emit()
    return true
  }

  closeFolder(): void {
    this.root = null
    this.rootName = ''
    this.cwd = '.'
    this.tree = null
    this.buffers = []
    this.activeId = null
    this.emit()
  }

  async refreshTree(): Promise<void> {
    if (!this.root) {
      this.tree = null
      this.emit()
      return
    }
    this.tree = await this.readDir(this.root, '.')
    this.emit()
  }

  private async readDir(dir: FileSystemDirectoryHandle, rel: string): Promise<FsNode> {
    const children: FsNode[] = []
    const dirAny = dir as FileSystemDirectoryHandle & {
      entries?: () => AsyncIterableIterator<[string, FileSystemHandle]>
      values?: () => AsyncIterableIterator<FileSystemHandle>
    }

    if (typeof dirAny.entries === 'function') {
      for await (const [name, handle] of dirAny.entries()) {
        await this.pushChild(children, name, handle, rel)
      }
    } else if (typeof dirAny.values === 'function') {
      for await (const handle of dirAny.values()) {
        await this.pushChild(children, handle.name, handle, rel)
      }
    }

    children.sort((a, b) => {
      if (a.kind !== b.kind) return a.kind === 'dir' ? -1 : 1
      return a.name.localeCompare(b.name)
    })
    return {
      name: rel === '.' ? this.rootName || dir.name : rel.split('/').pop()!,
      path: rel,
      kind: 'dir',
      handle: dir,
      children,
    }
  }

  private async pushChild(
    children: FsNode[],
    name: string,
    handle: FileSystemHandle,
    rel: string,
  ): Promise<void> {
    if (handle.kind === 'directory') {
      if (SKIP_DIRS.has(name)) return
      const childPath = rel === '.' ? name : `${rel}/${name}`
      children.push(await this.readDir(handle as FileSystemDirectoryHandle, childPath))
      return
    }
    if (TEXT_EXT.test(name) || name.endsWith('.ts')) {
      const childPath = rel === '.' ? name : `${rel}/${name}`
      children.push({ name, path: childPath, kind: 'file', handle })
    }
  }

  setCwd(path: string): void {
    this.cwd = path || '.'
    this.emit()
  }

  resolvePath(relative: string): string {
    if (!relative || relative === '.') return this.cwd
    if (relative.startsWith('/')) return relative.replace(/^\//, '')
    if (this.cwd === '.' || this.cwd === '') return relative
    const parts = [...this.cwd.split('/'), ...relative.split('/')]
    const out: string[] = []
    for (const p of parts) {
      if (!p || p === '.') continue
      if (p === '..') out.pop()
      else out.push(p)
    }
    return out.join('/') || '.'
  }

  async openFile(path: string): Promise<ProjectFileBuffer | null> {
    if (!this.root) return null
    const existing = this.buffers.find((b) => b.path === path)
    if (existing) {
      this.activeId = existing.id
      this.emit()
      return existing
    }
    const fileHandle = await this.getFileHandle(path)
    if (!fileHandle) return null
    const file = await fileHandle.getFile()
    const content = await file.text()
    const buf: ProjectFileBuffer = {
      id: uid('proj'),
      path,
      name: path.split('/').pop() ?? path,
      content,
      handle: fileHandle,
      dirty: false,
    }
    this.buffers = [...this.buffers, buf]
    this.activeId = buf.id
    this.emit()
    return buf
  }

  setActive(id: string): void {
    this.activeId = id
    this.emit()
  }

  updateContent(id: string, content: string): void {
    this.buffers = this.buffers.map((b) =>
      b.id === id ? { ...b, content, dirty: true } : b,
    )
    this.emit()
  }

  async saveActive(): Promise<void> {
    const active = this.getActive()
    if (!active?.handle) return
    const writable = await active.handle.createWritable()
    await writable.write(active.content)
    await writable.close()
    this.buffers = this.buffers.map((b) =>
      b.id === active.id ? { ...b, dirty: false } : b,
    )
    this.emit()
  }

  async writePath(path: string, content: string): Promise<void> {
    const handle = await this.getFileHandle(path, true)
    if (!handle) throw new Error(`Cannot write ${path}`)
    const writable = await handle.createWritable()
    await writable.write(content)
    await writable.close()
    this.buffers = this.buffers.map((b) =>
      b.path === path ? { ...b, content, dirty: false } : b,
    )
    this.emit()
  }

  async readText(path: string): Promise<string | null> {
    const handle = await this.getFileHandle(path)
    if (!handle) return null
    return (await handle.getFile()).text()
  }

  /** Flat list of text file paths under the project (for find-in-files). */
  listFilePaths(node: FsNode | null = this.tree): string[] {
    if (!node) return []
    if (node.kind === 'file') return [node.path]
    return (node.children ?? []).flatMap((c) => this.listFilePaths(c))
  }

  private async getFileHandle(
    path: string,
    create = false,
  ): Promise<FileSystemFileHandle | null> {
    if (!this.root || path === '.' || !path) return null
    const parts = path.split('/').filter(Boolean)
    let dir: FileSystemDirectoryHandle = this.root
    for (let i = 0; i < parts.length - 1; i++) {
      dir = await dir.getDirectoryHandle(parts[i]!, { create })
    }
    const name = parts[parts.length - 1]!
    try {
      return await dir.getFileHandle(name, { create })
    } catch {
      return null
    }
  }
}

export function supportsDirectoryPicker(): boolean {
  return typeof (window as DirPickerWindow).showDirectoryPicker === 'function'
}
