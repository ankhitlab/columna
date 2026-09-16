import { useEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'

export interface ExternalWindowProps {
  title: string
  open: boolean
  onClose: () => void
  children: ReactNode
  width?: number
  height?: number
}

/**
 * Spyder-style detached browser window that hosts React UI
 * and mirrors stylesheets from the main document.
 */
export function ExternalWindow({
  title,
  open,
  onClose,
  children,
  width = 1100,
  height = 720,
}: ExternalWindowProps) {
  const [container, setContainer] = useState<HTMLElement | null>(null)
  const popupRef = useRef<Window | null>(null)

  useEffect(() => {
    if (!open) {
      popupRef.current?.close()
      popupRef.current = null
      setContainer(null)
      return
    }

    const existing = popupRef.current
    if (existing && !existing.closed) {
      existing.document.title = title
      existing.focus()
      return
    }

    const left = Math.max(0, Math.round(window.screenX + (window.outerWidth - width) / 2))
    const top = Math.max(0, Math.round(window.screenY + (window.outerHeight - height) / 2))
    const features = `popup=yes,width=${width},height=${height},left=${left},top=${top}`
    const popup = window.open('', 'columna-studio-var-explorer', features)
    if (!popup) {
      onClose()
      return
    }

    popupRef.current = popup
    popup.document.title = title
    popup.document.body.innerHTML = ''
    popup.document.documentElement.lang = document.documentElement.lang || 'en'

    // Base styles so the blank document matches Studio before CSS clones apply
    const base = popup.document.createElement('style')
    base.textContent = `
      html, body, #studio-external-root {
        margin: 0;
        height: 100%;
        background: #f3efe6;
        color: #1c2430;
        font-family: 'Source Sans 3', 'Segoe UI', sans-serif;
      }
    `
    popup.document.head.appendChild(base)

    for (const node of Array.from(document.querySelectorAll('link[rel="stylesheet"], style'))) {
      popup.document.head.appendChild(node.cloneNode(true))
    }

    const root = popup.document.createElement('div')
    root.id = 'studio-external-root'
    popup.document.body.appendChild(root)
    setContainer(root)

    const onUnload = () => {
      popupRef.current = null
      setContainer(null)
      onClose()
    }
    popup.addEventListener('beforeunload', onUnload)

    return () => {
      popup.removeEventListener('beforeunload', onUnload)
      if (!popup.closed) popup.close()
      popupRef.current = null
      setContainer(null)
    }
    // Re-open only when `open` flips; title updates handled below
  }, [open])

  useEffect(() => {
    if (!open || !popupRef.current || popupRef.current.closed) return
    popupRef.current.document.title = title
  }, [open, title])

  if (!open || !container) return null
  return createPortal(children, container)
}
