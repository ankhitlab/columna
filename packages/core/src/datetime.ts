import { col } from './expr.js'
import type { Expr } from './expr.js'

/** Datetime helpers on epoch-millis columns (UTC). */
export const dt = {
  year(column: string): Expr {
    return col(column).dt.year()
  },
  month(column: string): Expr {
    return col(column).dt.month()
  },
  day(column: string): Expr {
    return col(column).dt.day()
  },
  hour(column: string): Expr {
    return col(column).dt.hour()
  },
  minute(column: string): Expr {
    return col(column).dt.minute()
  },
  second(column: string): Expr {
    return col(column).dt.second()
  },
  weekday(column: string): Expr {
    return col(column).dt.weekday()
  },
}

export function daysBetween(a: string, b: string): Expr {
  return col(a).sub(col(b)).div(86_400_000)
}
