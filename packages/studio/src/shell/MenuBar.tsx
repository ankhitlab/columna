import { useEffect, useRef, useState } from 'react'

export interface MenuAction {
  id: string
  label: string
  shortcut?: string
  disabled?: boolean
  separator?: boolean
}

export interface MenuDef {
  id: string
  label: string
  items: MenuAction[]
}

export interface MenuBarProps {
  menus: MenuDef[]
  onAction: (id: string) => void
}

export function MenuBar({ menus, onAction }: MenuBarProps) {
  const [openId, setOpenId] = useState<string | null>(null)
  const rootRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!openId) return
    const onDoc = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpenId(null)
    }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [openId])

  return (
    <div className="menubar" ref={rootRef}>
      {menus.map((menu) => (
        <div key={menu.id} className={`menu-root ${openId === menu.id ? 'open' : ''}`}>
          <button
            type="button"
            className="menu-trigger"
            onClick={() => setOpenId((cur) => (cur === menu.id ? null : menu.id))}
            onMouseEnter={() => {
              if (openId) setOpenId(menu.id)
            }}
          >
            {menu.label}
          </button>
          {openId === menu.id && (
            <div className="menu-dropdown">
              {menu.items.map((item, i) =>
                item.separator ? (
                  <div key={`sep-${i}`} className="menu-sep" />
                ) : (
                  <button
                    key={item.id}
                    type="button"
                    className="menu-item"
                    disabled={item.disabled}
                    onClick={() => {
                      setOpenId(null)
                      onAction(item.id)
                    }}
                  >
                    <span>{item.label}</span>
                    {item.shortcut && <span className="menu-shortcut">{item.shortcut}</span>}
                  </button>
                ),
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
