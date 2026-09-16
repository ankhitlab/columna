import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { PowerQueryStudio } from './power-query/PowerQueryStudio'
import './styles/studio.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <PowerQueryStudio />
  </StrictMode>,
)
