#![deny(clippy::all)]

use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;

#[inline]
fn cmp_f64(op: u32, v: f64, lit: f64) -> bool {
  match op {
    0 => v == lit,
    1 => v != lit,
    2 => v > lit,
    3 => v >= lit,
    4 => v < lit,
    5 => v <= lit,
    _ => false,
  }
}

#[inline]
fn cmp_i32(op: u32, v: i32, lit: i32) -> bool {
  match op {
    0 => v == lit,
    1 => v != lit,
    2 => v > lit,
    3 => v >= lit,
    4 => v < lit,
    5 => v <= lit,
    _ => false,
  }
}

fn chunk_size(n: usize) -> usize {
  ((n / rayon::current_num_threads().max(1)) + 1).max(16_384)
}

/// Parallel dual filter: Int32 AND Float64 (bench: age + salary). Zero-copy views + rayon.
#[napi]
pub fn filter_and2_i32_f64(
  a: Int32Array,
  b: Float64Array,
  op_a: u32,
  lit_a: f64,
  op_b: u32,
  lit_b: f64,
) -> Uint32Array {
  let n = a.len().min(b.len());
  let av = a.as_ref();
  let bv = b.as_ref();
  let lit_ai = lit_a as i32;
  let chunk = chunk_size(n);

  let parts: Vec<Vec<u32>> = av[..n]
    .par_chunks(chunk)
    .enumerate()
    .map(|(ci, a_chunk)| {
      let base = ci * chunk;
      let b_chunk = &bv[base..base + a_chunk.len()];
      let mut local = Vec::with_capacity(a_chunk.len() / 2);
      for (j, (&va, &vb)) in a_chunk.iter().zip(b_chunk.iter()).enumerate() {
        if cmp_i32(op_a, va, lit_ai) && cmp_f64(op_b, vb, lit_b) {
          local.push((base + j) as u32);
        }
      }
      local
    })
    .collect();

  let total: usize = parts.iter().map(|p| p.len()).sum();
  let mut out = Vec::with_capacity(total);
  for p in parts {
    out.extend_from_slice(&p);
  }
  Uint32Array::new(out)
}

/// Parallel dual filter: two Int32 columns.
#[napi]
pub fn filter_and2_i32_i32(
  a: Int32Array,
  b: Int32Array,
  op_a: u32,
  lit_a: f64,
  op_b: u32,
  lit_b: f64,
) -> Uint32Array {
  let n = a.len().min(b.len());
  let av = a.as_ref();
  let bv = b.as_ref();
  let lit_ai = lit_a as i32;
  let lit_bi = lit_b as i32;
  let chunk = chunk_size(n);

  let parts: Vec<Vec<u32>> = av[..n]
    .par_chunks(chunk)
    .enumerate()
    .map(|(ci, a_chunk)| {
      let base = ci * chunk;
      let b_chunk = &bv[base..base + a_chunk.len()];
      let mut local = Vec::with_capacity(a_chunk.len() / 2);
      for (j, (&va, &vb)) in a_chunk.iter().zip(b_chunk.iter()).enumerate() {
        if cmp_i32(op_a, va, lit_ai) && cmp_i32(op_b, vb, lit_bi) {
          local.push((base + j) as u32);
        }
      }
      local
    })
    .collect();

  let total: usize = parts.iter().map(|p| p.len()).sum();
  let mut out = Vec::with_capacity(total);
  for p in parts {
    out.extend_from_slice(&p);
  }
  Uint32Array::new(out)
}

/// Parallel dual filter: two Float64 columns with arbitrary cmp ops.
#[napi]
pub fn filter_and2_f64_f64(
  a: Float64Array,
  b: Float64Array,
  op_a: u32,
  lit_a: f64,
  op_b: u32,
  lit_b: f64,
) -> Uint32Array {
  let n = a.len().min(b.len());
  let av = a.as_ref();
  let bv = b.as_ref();
  let chunk = chunk_size(n);

  let parts: Vec<Vec<u32>> = av[..n]
    .par_chunks(chunk)
    .enumerate()
    .map(|(ci, a_chunk)| {
      let base = ci * chunk;
      let b_chunk = &bv[base..base + a_chunk.len()];
      let mut local = Vec::with_capacity(a_chunk.len() / 2);
      for (j, (&va, &vb)) in a_chunk.iter().zip(b_chunk.iter()).enumerate() {
        if cmp_f64(op_a, va, lit_a) && cmp_f64(op_b, vb, lit_b) {
          local.push((base + j) as u32);
        }
      }
      local
    })
    .collect();

  let total: usize = parts.iter().map(|p| p.len()).sum();
  let mut out = Vec::with_capacity(total);
  for p in parts {
    out.extend_from_slice(&p);
  }
  Uint32Array::new(out)
}

/// Argsort f64 values (nulls encoded as NaN are sorted last when `nulls_last`).
/// Returns row indices in sorted order. Stable-ish via index tie-break.
#[napi]
pub fn argsort_f64(values: Float64Array, descending: bool, nulls_last: bool) -> Uint32Array {
  let n = values.len();
  let data = values.as_ref();
  let mut idx: Vec<u32> = (0..n as u32).collect();
  idx.par_sort_by(|&ia, &ib| {
    let a = data[ia as usize];
    let b = data[ib as usize];
    let a_nan = a.is_nan();
    let b_nan = b.is_nan();
    if a_nan || b_nan {
      return match (a_nan, b_nan) {
        (true, true) => ia.cmp(&ib),
        (true, false) => {
          if nulls_last {
            std::cmp::Ordering::Greater
          } else {
            std::cmp::Ordering::Less
          }
        }
        (false, true) => {
          if nulls_last {
            std::cmp::Ordering::Less
          } else {
            std::cmp::Ordering::Greater
          }
        }
        _ => unreachable!(),
      };
    }
    let ord = a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal);
    let ord = if descending { ord.reverse() } else { ord };
    if ord == std::cmp::Ordering::Equal {
      ia.cmp(&ib)
    } else {
      ord
    }
  });
  Uint32Array::new(idx)
}

/// Argsort i32 values. `null_bitmap` optional packed validity (1 bit per row); when absent all valid.
#[napi]
pub fn argsort_i32(
  values: Int32Array,
  descending: bool,
  nulls_last: bool,
  null_bitmap: Option<Uint8Array>,
) -> Uint32Array {
  let n = values.len();
  let data = values.as_ref();
  let bm = null_bitmap.as_ref().map(|b| b.as_ref());
  let is_null = |i: usize| -> bool {
    match bm {
      Some(bits) => {
        let byte = i / 8;
        if byte >= bits.len() {
          return true;
        }
        (bits[byte] & (1 << (i % 8))) == 0
      }
      None => false,
    }
  };
  let mut idx: Vec<u32> = (0..n as u32).collect();
  idx.par_sort_by(|&ia, &ib| {
    let a_null = is_null(ia as usize);
    let b_null = is_null(ib as usize);
    if a_null || b_null {
      return match (a_null, b_null) {
        (true, true) => ia.cmp(&ib),
        (true, false) => {
          if nulls_last {
            std::cmp::Ordering::Greater
          } else {
            std::cmp::Ordering::Less
          }
        }
        (false, true) => {
          if nulls_last {
            std::cmp::Ordering::Less
          } else {
            std::cmp::Ordering::Greater
          }
        }
        _ => unreachable!(),
      };
    }
    let a = data[ia as usize];
    let b = data[ib as usize];
    let ord = a.cmp(&b);
    let ord = if descending { ord.reverse() } else { ord };
    if ord == std::cmp::Ordering::Equal {
      ia.cmp(&ib)
    } else {
      ord
    }
  });
  Uint32Array::new(idx)
}

/// Parallel dense groupby min/max for f64 columns (same layout as groupby_sums_f64).
#[napi(object)]
pub struct GroupMinMaxOut {
  pub mins: Float64Array,
  pub maxs: Float64Array,
  pub counts: Float64Array,
  pub used: Uint8Array,
}

#[napi]
pub fn groupby_minmax_f64(codes: Uint32Array, card: u32, cols: Vec<Float64Array>) -> Result<GroupMinMaxOut> {
  let card = card as usize;
  if card == 0 {
    return Ok(GroupMinMaxOut {
      mins: Float64Array::new(vec![]),
      maxs: Float64Array::new(vec![]),
      counts: Float64Array::new(vec![]),
      used: Uint8Array::new(vec![]),
    });
  }
  let n = codes.len();
  let ncols = cols.len();
  let code_ref = codes.as_ref();
  let col_refs: Vec<&[f64]> = cols.iter().map(|c| c.as_ref()).collect();
  for c in &col_refs {
    if c.len() < n {
      return Err(Error::from_reason("groupby_minmax_f64: column shorter than codes"));
    }
  }

  let chunk = chunk_size(n);
  let (mins, maxs, counts) = code_ref[..n]
    .par_chunks(chunk)
    .enumerate()
    .map(|(ci, code_chunk)| {
      let base = ci * chunk;
      let mut mins = vec![f64::INFINITY; ncols * card];
      let mut maxs = vec![f64::NEG_INFINITY; ncols * card];
      let mut counts = vec![0f64; card];
      for (j, &code) in code_chunk.iter().enumerate() {
        let c = code as usize;
        if c >= card {
          continue;
        }
        counts[c] += 1.0;
        let row = base + j;
        for (ci, col) in col_refs.iter().enumerate() {
          let v = col[row];
          let ix = ci * card + c;
          if v < mins[ix] {
            mins[ix] = v;
          }
          if v > maxs[ix] {
            maxs[ix] = v;
          }
        }
      }
      (mins, maxs, counts)
    })
    .reduce(
      || (
        vec![f64::INFINITY; ncols * card],
        vec![f64::NEG_INFINITY; ncols * card],
        vec![0f64; card],
      ),
      |mut a, b| {
        for i in 0..a.0.len() {
          if b.0[i] < a.0[i] {
            a.0[i] = b.0[i];
          }
          if b.1[i] > a.1[i] {
            a.1[i] = b.1[i];
          }
        }
        for i in 0..card {
          a.2[i] += b.2[i];
        }
        a
      },
    );

  let used: Vec<u8> = counts.iter().map(|&c| if c > 0.0 { 1 } else { 0 }).collect();
  Ok(GroupMinMaxOut {
    mins: Float64Array::new(mins),
    maxs: Float64Array::new(maxs),
    counts: Float64Array::new(counts),
    used: Uint8Array::new(used),
  })
}

/// Parallel gather of f64 values by indices.
#[napi]
pub fn gather_f64(src: Float64Array, indices: Uint32Array) -> Float64Array {
  let s = src.as_ref();
  let idx = indices.as_ref();
  let mut out = vec![0f64; idx.len()];
  out.par_iter_mut().zip(idx.par_iter()).for_each(|(slot, &i)| {
    let i = i as usize;
    *slot = if i < s.len() { s[i] } else { f64::NAN };
  });
  Float64Array::new(out)
}

/// Parallel gather of i32 values by indices.
#[napi]
pub fn gather_i32(src: Int32Array, indices: Uint32Array) -> Int32Array {
  let s = src.as_ref();
  let idx = indices.as_ref();
  let mut out = vec![0i32; idx.len()];
  out.par_iter_mut().zip(idx.par_iter()).for_each(|(slot, &i)| {
    let i = i as usize;
    *slot = if i < s.len() { s[i] } else { 0 };
  });
  Int32Array::new(out)
}

/// Dense equi-join probe: `out[i] = dense[left[i] - r_min]` or -1 if OOB / empty slot.
#[napi]
pub fn join_probe_dense_i32(left_keys: Int32Array, dense: Int32Array, r_min: i32) -> Int32Array {
  let keys = left_keys.as_ref();
  let table = dense.as_ref();
  let span = table.len() as i32;
  let mut out = vec![-1i32; keys.len()];
  out.par_iter_mut().zip(keys.par_iter()).for_each(|(slot, &k)| {
    let off = k.wrapping_sub(r_min);
    if off >= 0 && off < span {
      *slot = table[off as usize];
    }
  });
  Int32Array::new(out)
}

/// Semi/anti filter via dense probe. `want_hit=true` → semi, false → anti.
#[napi]
pub fn join_semi_dense_i32(
  left_keys: Int32Array,
  dense: Int32Array,
  r_min: i32,
  want_hit: bool,
) -> Uint32Array {
  let keys = left_keys.as_ref();
  let table = dense.as_ref();
  let span = table.len() as i32;
  let chunk = chunk_size(keys.len());

  let parts: Vec<Vec<u32>> = keys
    .par_chunks(chunk)
    .enumerate()
    .map(|(ci, chunk_keys)| {
      let base = ci * chunk;
      let mut local = Vec::with_capacity(chunk_keys.len());
      for (j, &k) in chunk_keys.iter().enumerate() {
        let off = k.wrapping_sub(r_min);
        let hit = off >= 0 && off < span && table[off as usize] >= 0;
        if hit == want_hit {
          local.push((base + j) as u32);
        }
      }
      local
    })
    .collect();

  let total: usize = parts.iter().map(|p| p.len()).sum();
  let mut out = Vec::with_capacity(total);
  for p in parts {
    out.extend_from_slice(&p);
  }
  Uint32Array::new(out)
}

/// Generic multi-column AND filter on f64 columns.
/// `ops`: 0=eq 1=neq 2=gt 3=gte 4=lt 5=lte. `lits`: literal per column.
/// Returns matching row indices.
#[napi]
pub fn filter_f64(cols: Vec<Float64Array>, ops: Vec<u32>, lits: Vec<f64>) -> Result<Uint32Array> {
  let ncols = cols.len();
  if ncols == 0 {
    return Ok(Uint32Array::new(vec![]));
  }
  if ops.len() != ncols || lits.len() != ncols {
    return Err(Error::from_reason("filter_f64: ops/lits length mismatch"));
  }
  let n = cols[0].len();
  let col_refs: Vec<&[f64]> = cols.iter().map(|c| c.as_ref()).collect();
  for c in &col_refs {
    if c.len() < n {
      return Err(Error::from_reason("filter_f64: column shorter than first"));
    }
  }
  let chunk = chunk_size(n);

  let parts: Vec<Vec<u32>> = (0..n)
    .into_par_iter()
    .chunks(chunk)
    .map(|rows| {
      let mut local = Vec::with_capacity(rows.len() / 2);
      for i in rows {
        let row = i;
        let mut hit = true;
        for c in 0..ncols {
          if !cmp_f64(ops[c], col_refs[c][row], lits[c]) {
            hit = false;
            break;
          }
        }
        if hit {
          local.push(row as u32);
        }
      }
      local
    })
    .collect();

  let total: usize = parts.iter().map(|p| p.len()).sum();
  let mut out = Vec::with_capacity(total);
  for p in parts {
    out.extend_from_slice(&p);
  }
  Ok(Uint32Array::new(out))
}

/// Lexicographic multi-key argsort on f64 columns (NaN = null, sorted per `nulls_last[k]`).
/// `descending` / `nulls_last` must each have length == keys.len(). Stable via index tie-break.
#[napi]
pub fn argsort_multi_f64(
  keys: Vec<Float64Array>,
  descending: Vec<bool>,
  nulls_last: Vec<bool>,
) -> Result<Uint32Array> {
  let nkeys = keys.len();
  if nkeys == 0 {
    return Ok(Uint32Array::new(vec![]));
  }
  if descending.len() != nkeys || nulls_last.len() != nkeys {
    return Err(Error::from_reason("argsort_multi_f64: flag length mismatch"));
  }
  let n = keys[0].len();
  let key_refs: Vec<&[f64]> = keys.iter().map(|c| c.as_ref()).collect();
  for c in &key_refs {
    if c.len() < n {
      return Err(Error::from_reason("argsort_multi_f64: key shorter than first"));
    }
  }
  let mut idx: Vec<u32> = (0..n as u32).collect();
  idx.par_sort_by(|&ia, &ib| {
    for k in 0..nkeys {
      let a = key_refs[k][ia as usize];
      let b = key_refs[k][ib as usize];
      let a_nan = a.is_nan();
      let b_nan = b.is_nan();
      if a_nan || b_nan {
        return match (a_nan, b_nan) {
          (true, true) => ia.cmp(&ib),
          (true, false) => {
            if nulls_last[k] {
              std::cmp::Ordering::Greater
            } else {
              std::cmp::Ordering::Less
            }
          }
          (false, true) => {
            if nulls_last[k] {
              std::cmp::Ordering::Less
            } else {
              std::cmp::Ordering::Greater
            }
          }
          _ => unreachable!(),
        };
      }
      let ord = a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal);
      let ord = if descending[k] { ord.reverse() } else { ord };
      if ord != std::cmp::Ordering::Equal {
        return ord;
      }
    }
    ia.cmp(&ib)
  });
  Ok(Uint32Array::new(idx))
}

/// Parallel unique (first-seen) over f64 columns. Returns first row index of each distinct tuple.
#[napi]
pub fn unique_f64(cols: Vec<Float64Array>) -> Result<Uint32Array> {
  let ncols = cols.len();
  if ncols == 0 {
    return Ok(Uint32Array::new(vec![]));
  }
  let n = cols[0].len();
  let col_refs: Vec<&[f64]> = cols.iter().map(|c| c.as_ref()).collect();
  for c in &col_refs {
    if c.len() < n {
      return Err(Error::from_reason("unique_f64: column shorter than first"));
    }
  }
  let chunk = chunk_size(n);

  // Each chunk builds its own first-seen index set; main merges keeping global first-seen.
  let parts: Vec<Vec<u32>> = (0..n)
    .into_par_iter()
    .chunks(chunk)
    .map(|rows| {
      let mut seen: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
      let mut local = Vec::new();
      let mut keybuf = vec![0u8; ncols * 8];
      for i in rows {
        for c in 0..ncols {
          keybuf[c * 8..c * 8 + 8].copy_from_slice(&col_refs[c][i].to_le_bytes());
        }
        if seen.insert(keybuf.clone()) {
          local.push(i as u32);
        }
      }
      local
    })
    .collect();

  // Merge: keep first-seen across chunks (chunks are in row order, so earlier chunks win).
  let mut global: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
  let mut out = Vec::new();
  let mut keybuf = vec![0u8; ncols * 8];
  for part in parts {
    for &i in part.iter() {
      for c in 0..ncols {
        keybuf[c * 8..c * 8 + 8].copy_from_slice(&col_refs[c][i as usize].to_le_bytes());
      }
      if global.insert(keybuf.clone()) {
        out.push(i);
      }
    }
  }
  Ok(Uint32Array::new(out))
}

/// Hash-join build side: construct a dense probe table from right int32 keys.
/// `dense[k - r_min]` = first right row index with that key, or -1 when the slot is empty.
/// Pairs with `join_probe_dense_i32` / `join_semi_dense_i32`.
#[napi]
pub fn join_build_dense_i32(right_keys: Int32Array, r_min: i32) -> Result<Int32Array> {
  let keys = right_keys.as_ref();
  let n = keys.len();
  if n == 0 {
    return Ok(Int32Array::new(vec![]));
  }
  let mut r_max = r_min;
  for &k in keys.iter() {
    if k > r_max {
      r_max = k;
    }
  }
  if r_max < r_min {
    return Err(Error::from_reason("join_build_dense_i32: r_max < r_min"));
  }
  let span = (r_max - r_min + 1) as usize;
  let mut dense = vec![-1i32; span];
  for (i, &k) in keys.iter().enumerate() {
    let off = (k - r_min) as usize;
    // Last-wins to match the JS dense build semantics (one match per key).
    dense[off] = i as i32;
  }
  Ok(Int32Array::new(dense))
}

#[napi(object)]
pub struct GroupSumsOut {
  pub sums: Float64Array,
  pub counts: Float64Array,
  pub used: Uint8Array,
}

/// Parallel dense groupby: one category-code stream + many f64 value columns.
/// Returns concatenated sums `[col0_card | col1_card | ...]`, per-group counts, and used flags.
#[napi]
pub fn groupby_sums_f64(codes: Uint32Array, card: u32, cols: Vec<Float64Array>) -> Result<GroupSumsOut> {
  let card = card as usize;
  if card == 0 {
    return Ok(GroupSumsOut {
      sums: Float64Array::new(vec![]),
      counts: Float64Array::new(vec![]),
      used: Uint8Array::new(vec![]),
    });
  }
  let n = codes.len();
  let ncols = cols.len();
  let code_ref = codes.as_ref();
  let col_refs: Vec<&[f64]> = cols.iter().map(|c| c.as_ref()).collect();
  for c in &col_refs {
    if c.len() < n {
      return Err(Error::from_reason("groupby_sums_f64: column shorter than codes"));
    }
  }

  let chunk = chunk_size(n);
  let (sums, counts) = code_ref[..n]
    .par_chunks(chunk)
    .enumerate()
    .map(|(ci, code_chunk)| {
      let base = ci * chunk;
      let mut sums = vec![0f64; ncols * card];
      let mut counts = vec![0f64; card];
      for (j, &code) in code_chunk.iter().enumerate() {
        let c = code as usize;
        if c >= card {
          continue;
        }
        counts[c] += 1.0;
        let row = base + j;
        for (ci, col) in col_refs.iter().enumerate() {
          sums[ci * card + c] += col[row];
        }
      }
      (sums, counts)
    })
    .reduce(
      || (vec![0f64; ncols * card], vec![0f64; card]),
      |mut a, b| {
        for i in 0..a.0.len() {
          a.0[i] += b.0[i];
        }
        for i in 0..card {
          a.1[i] += b.1[i];
        }
        a
      },
    );

  let used: Vec<u8> = counts.iter().map(|&c| if c > 0.0 { 1 } else { 0 }).collect();
  Ok(GroupSumsOut {
    sums: Float64Array::new(sums),
    counts: Float64Array::new(counts),
    used: Uint8Array::new(used),
  })
}

/// Parallel string contains → uint8 mask (1 = hit). Copies strings once into Rust.
#[napi]
pub fn str_contains(values: Vec<String>, needle: String) -> Uint8Array {
  let mask: Vec<u8> = values
    .par_iter()
    .map(|s| if s.contains(&needle) { 1 } else { 0 })
    .collect();
  Uint8Array::new(mask)
}

/// Parallel ASCII/Unicode lower — returns new strings (for utf8 materialize).
#[napi]
pub fn str_to_lower(values: Vec<String>) -> Vec<String> {
  values.par_iter().map(|s| s.to_lowercase()).collect()
}

// ---- CSV (unquoted, default locale) ----------------------------------------------------------------

#[napi(object)]
pub struct NativeCsvColumn {
  pub name: String,
  /// "i32" | "f64" | "bool" | "category" | "utf8"
  pub dtype: String,
  pub null_bitmap: Option<Uint8Array>,
  pub f64_data: Option<Float64Array>,
  pub i32_data: Option<Int32Array>,
  pub bool_data: Option<Uint8Array>,
  pub cat_codes: Option<Uint32Array>,
  pub dictionary: Option<Vec<String>>,
  pub utf8_data: Option<Vec<String>>,
}

#[napi(object)]
pub struct NativeCsvTable {
  pub num_rows: u32,
  pub columns: Vec<NativeCsvColumn>,
}

/// Column payload for native CSV write (mirrors NativeCsvColumn).
#[napi(object)]
pub struct NativeCsvWriteColumn {
  pub name: String,
  pub dtype: String,
  pub null_bitmap: Option<Uint8Array>,
  pub f64_data: Option<Float64Array>,
  pub i32_data: Option<Int32Array>,
  pub bool_data: Option<Uint8Array>,
  pub cat_codes: Option<Uint32Array>,
  pub dictionary: Option<Vec<String>>,
  pub utf8_data: Option<Vec<String>>,
}

/// Online typed builder — no per-cell `Cell` / String for numerics (matches JS ColumnBuilder).
struct ColAcc {
  name: String,
  /// Numeric values, or dictionary codes once in text mode.
  vals: Vec<f64>,
  /// One byte per row: 1 = valid, 0 = null (packed at finish).
  valid: Vec<u8>,
  saw_int: bool,
  saw_float: bool,
  saw_bool: bool,
  int_outside_i32: bool,
  any_null: bool,
  dict: Option<Vec<String>>,
  codes_map: Option<std::collections::HashMap<String, u32>>,
}

impl ColAcc {
  fn new(name: String) -> Self {
    Self {
      name,
      vals: Vec::new(),
      valid: Vec::new(),
      saw_int: false,
      saw_float: false,
      saw_bool: false,
      int_outside_i32: false,
      any_null: false,
      dict: None,
      codes_map: None,
    }
  }

  fn with_capacity(name: String, cap: usize) -> Self {
    let mut c = Self::new(name);
    c.vals.reserve(cap);
    c.valid.reserve(cap);
    c
  }

  #[inline]
  fn push_null(&mut self) {
    self.vals.push(0.0);
    self.valid.push(0);
    self.any_null = true;
  }

  #[inline]
  fn push_num(&mut self, v: f64) {
    if self.codes_map.is_some() {
      let s = Self::num_to_string(v, self.saw_bool, self.saw_int, self.saw_float);
      let code = self.code(s);
      self.vals.push(code as f64);
      self.valid.push(1);
      return;
    }
    self.vals.push(v);
    self.valid.push(1);
    if v.fract() == 0.0 && v.is_finite() {
      self.saw_int = true;
      if v > 2147483647.0 || v < -2147483648.0 {
        self.int_outside_i32 = true;
      }
    } else {
      self.saw_float = true;
    }
  }

  #[inline]
  fn push_bool(&mut self, b: bool) {
    if self.codes_map.is_some() {
      let code = self.code(if b { "true".into() } else { "false".into() });
      self.vals.push(code as f64);
      self.valid.push(1);
      return;
    }
    self.vals.push(if b { 1.0 } else { 0.0 });
    self.valid.push(1);
    self.saw_bool = true;
  }

  fn push_text(&mut self, s: String) {
    if self.codes_map.is_none() {
      self.enter_text_mode();
    }
    let code = self.code(s);
    self.vals.push(code as f64);
    self.valid.push(1);
  }

  fn enter_text_mode(&mut self) {
    if self.codes_map.is_some() {
      return;
    }
    self.dict = Some(Vec::new());
    self.codes_map = Some(std::collections::HashMap::new());
    let n = self.vals.len();
    let saw_bool = self.saw_bool;
    let saw_int = self.saw_int;
    let saw_float = self.saw_float;
    for i in 0..n {
      if self.valid[i] == 0 {
        continue;
      }
      let s = Self::num_to_string(self.vals[i], saw_bool, saw_int, saw_float);
      let code = {
        // inline code() without reborrow issues
        let map = self.codes_map.as_mut().unwrap();
        let dict = self.dict.as_mut().unwrap();
        if let Some(&c) = map.get(&s) {
          c
        } else {
          let c = dict.len() as u32;
          map.insert(s.clone(), c);
          dict.push(s);
          c
        }
      };
      self.vals[i] = code as f64;
    }
  }

  fn code(&mut self, s: String) -> u32 {
    let map = self.codes_map.as_mut().unwrap();
    let dict = self.dict.as_mut().unwrap();
    if let Some(&c) = map.get(&s) {
      return c;
    }
    let c = dict.len() as u32;
    map.insert(s.clone(), c);
    dict.push(s);
    c
  }

  fn num_to_string(x: f64, saw_bool: bool, saw_int: bool, saw_float: bool) -> String {
    if saw_bool && !saw_int && !saw_float {
      return if x != 0.0 { "true".into() } else { "false".into() };
    }
    // Match JS String(number): integers without ".0"
    if x.fract() == 0.0 && x.is_finite() && x.abs() <= (1u64 << 53) as f64 {
      format!("{}", x as i64)
    } else {
      let mut buf = ryu::Buffer::new();
      buf.format(x).to_string()
    }
  }

  fn append_merge(&mut self, mut other: ColAcc) {
    self.saw_int |= other.saw_int;
    self.saw_float |= other.saw_float;
    self.saw_bool |= other.saw_bool;
    self.int_outside_i32 |= other.int_outside_i32;
    self.any_null |= other.any_null;

    let self_text = self.codes_map.is_some();
    let other_text = other.codes_map.is_some();

    if self_text || other_text {
      if !self_text {
        self.enter_text_mode();
      }
      if !other_text {
        other.enter_text_mode();
      }
      // Remap other's codes into self's dictionary.
      let other_dict = other.dict.take().unwrap_or_default();
      let mut remap = vec![0u32; other_dict.len()];
      for (i, s) in other_dict.into_iter().enumerate() {
        remap[i] = self.code(s);
      }
      for i in 0..other.vals.len() {
        if other.valid[i] != 0 {
          let old = other.vals[i] as u32;
          other.vals[i] = remap[old as usize] as f64;
        }
      }
    }

    self.vals.append(&mut other.vals);
    self.valid.append(&mut other.valid);
  }

  fn finish(self) -> NativeCsvColumn {
    let n = self.vals.len();
    let null_bitmap = if self.any_null {
      let mut bm = vec![0u8; (n + 7) / 8];
      for (i, &v) in self.valid.iter().enumerate() {
        if v != 0 {
          bm[i >> 3] |= 1 << (i & 7);
        }
      }
      Some(Uint8Array::new(bm))
    } else {
      None
    };

    if let Some(dict) = self.dict {
      if dict.len() <= (1024usize).max(n / 4) {
        let mut codes = vec![0u32; n];
        for i in 0..n {
          codes[i] = self.vals[i] as u32;
        }
        return NativeCsvColumn {
          name: self.name,
          dtype: "category".into(),
          null_bitmap,
          f64_data: None,
          i32_data: None,
          bool_data: None,
          cat_codes: Some(Uint32Array::new(codes)),
          dictionary: Some(dict),
          utf8_data: None,
        };
      }
      let mut utf8 = vec![String::new(); n];
      for i in 0..n {
        if self.valid[i] != 0 {
          utf8[i] = dict[self.vals[i] as usize].clone();
        }
      }
      return NativeCsvColumn {
        name: self.name,
        dtype: "utf8".into(),
        null_bitmap,
        f64_data: None,
        i32_data: None,
        bool_data: None,
        cat_codes: None,
        dictionary: None,
        utf8_data: Some(utf8),
      };
    }

    if self.saw_float || self.int_outside_i32 || (!self.saw_int && !self.saw_bool) {
      return NativeCsvColumn {
        name: self.name,
        dtype: "f64".into(),
        null_bitmap,
        f64_data: Some(Float64Array::new(self.vals)),
        i32_data: None,
        bool_data: None,
        cat_codes: None,
        dictionary: None,
        utf8_data: None,
      };
    }

    if self.saw_int {
      let mut data = vec![0i32; n];
      for i in 0..n {
        if self.valid[i] != 0 {
          data[i] = self.vals[i] as i32;
        }
      }
      return NativeCsvColumn {
        name: self.name,
        dtype: "i32".into(),
        null_bitmap,
        f64_data: None,
        i32_data: Some(Int32Array::new(data)),
        bool_data: None,
        cat_codes: None,
        dictionary: None,
        utf8_data: None,
      };
    }

    if self.saw_bool {
      let mut data = vec![0u8; n];
      for i in 0..n {
        if self.valid[i] != 0 {
          data[i] = if self.vals[i] != 0.0 { 1 } else { 0 };
        }
      }
      return NativeCsvColumn {
        name: self.name,
        dtype: "bool".into(),
        null_bitmap,
        f64_data: None,
        i32_data: None,
        bool_data: Some(Uint8Array::new(data)),
        cat_codes: None,
        dictionary: None,
        utf8_data: None,
      };
    }

    NativeCsvColumn {
      name: self.name,
      dtype: "f64".into(),
      null_bitmap,
      f64_data: Some(Float64Array::new(vec![0f64; n])),
      i32_data: None,
      bool_data: None,
      cat_codes: None,
      dictionary: None,
      utf8_data: None,
    }
  }
}

fn is_null_token(s: &[u8]) -> bool {
  match s {
    b"" | b"null" | b"NULL" | b"Null" | b"na" | b"NA" | b"Na" | b"nA" | b"nan" | b"NaN" | b"NAN" | b"Nan" => true,
    _ => {
      if s.len() == 2 {
        let a = s[0].to_ascii_lowercase();
        let b = s[1].to_ascii_lowercase();
        return a == b'n' && b == b'a';
      }
      if s.len() == 3 {
        let a = s[0].to_ascii_lowercase();
        let b = s[1].to_ascii_lowercase();
        let c = s[2].to_ascii_lowercase();
        return a == b'n' && b == b'a' && c == b'n';
      }
      if s.len() == 4 {
        let a = s[0].to_ascii_lowercase();
        let b = s[1].to_ascii_lowercase();
        let c = s[2].to_ascii_lowercase();
        let d = s[3].to_ascii_lowercase();
        return a == b'n' && b == b'u' && c == b'l' && d == b'l';
      }
      false
    }
  }
}

fn is_bool_token(s: &[u8]) -> Option<bool> {
  if s.len() == 4 {
    let a = s[0].to_ascii_lowercase();
    let b = s[1].to_ascii_lowercase();
    let c = s[2].to_ascii_lowercase();
    let d = s[3].to_ascii_lowercase();
    if a == b't' && b == b'r' && c == b'u' && d == b'e' {
      return Some(true);
    }
  }
  if s.len() == 5 {
    let a = s[0].to_ascii_lowercase();
    let b = s[1].to_ascii_lowercase();
    let c = s[2].to_ascii_lowercase();
    let d = s[3].to_ascii_lowercase();
    let e = s[4].to_ascii_lowercase();
    if a == b'f' && b == b'a' && c == b'l' && d == b's' && e == b'e' {
      return Some(false);
    }
  }
  None
}

/// Parsed field without allocating for the common numeric/bool/null cases.
enum ParsedField {
  Null,
  Bool(bool),
  Num(f64),
  /// Rare: oversized integer digits kept as text.
  BigInt(String),
  Text(String),
}

fn parse_number(s: &[u8]) -> Option<ParsedField> {
  if s.is_empty() {
    return None;
  }
  let mut i = 0usize;
  let mut sign = 1f64;
  if s[0] == b'+' {
    i = 1;
  } else if s[0] == b'-' {
    sign = -1.0;
    i = 1;
  }
  if i >= s.len() {
    return None;
  }
  let mut int_digits = 0usize;
  let mut int_val = 0f64;
  while i < s.len() && s[i].is_ascii_digit() {
    int_digits += 1;
    int_val = int_val * 10.0 + (s[i] - b'0') as f64;
    i += 1;
  }
  if int_digits == 0 {
    return None;
  }
  if i >= s.len() {
    if int_digits > 15 {
      let token = std::str::from_utf8(s).ok()?.to_string();
      let n: f64 = token.parse().ok()?;
      if n.fract() == 0.0 && n.abs() <= (1u64 << 53) as f64 {
        return Some(ParsedField::Num(n));
      }
      return Some(ParsedField::BigInt(token));
    }
    let n = sign * int_val;
    if n.abs() <= (1u64 << 53) as f64 {
      return Some(ParsedField::Num(n));
    }
    return Some(ParsedField::BigInt(std::str::from_utf8(s).ok()?.to_string()));
  }
  if s[i] == b'.' {
    i += 1;
    let mut frac_digits = 0usize;
    let mut frac_val = 0f64;
    let mut scale = 1f64;
    while i < s.len() && s[i].is_ascii_digit() {
      frac_digits += 1;
      frac_val = frac_val * 10.0 + (s[i] - b'0') as f64;
      scale *= 10.0;
      i += 1;
    }
    if frac_digits == 0 {
      return None;
    }
    if i >= s.len() {
      return Some(ParsedField::Num(sign * (int_val + frac_val / scale)));
    }
  }
  if i < s.len() && (s[i] == b'e' || s[i] == b'E') {
    let token = std::str::from_utf8(s).ok()?;
    let n: f64 = token.parse().ok()?;
    return Some(ParsedField::Num(n));
  }
  None
}

fn parse_field(raw: &[u8]) -> ParsedField {
  let mut start = 0usize;
  let mut end = raw.len();
  while start < end && raw[start] <= b' ' {
    start += 1;
  }
  while end > start && raw[end - 1] <= b' ' {
    end -= 1;
  }
  let s = &raw[start..end];
  if s.is_empty() || is_null_token(s) {
    return ParsedField::Null;
  }
  if let Some(b) = is_bool_token(s) {
    return ParsedField::Bool(b);
  }
  if let Some(c) = parse_number(s) {
    return c;
  }
  ParsedField::Text(String::from_utf8_lossy(raw).into_owned())
}

#[inline]
fn push_parsed(col: &mut ColAcc, field: ParsedField) {
  match field {
    ParsedField::Null => col.push_null(),
    ParsedField::Bool(b) => col.push_bool(b),
    ParsedField::Num(v) => col.push_num(v),
    ParsedField::BigInt(s) | ParsedField::Text(s) => col.push_text(s),
  }
}

/// Scan fields without allocating a Vec per row.
fn for_each_field(line: &[u8], delim: u8, mut f: impl FnMut(usize, &[u8])) {
  let mut start = 0usize;
  let mut ci = 0usize;
  for (i, &b) in line.iter().enumerate() {
    if b == delim {
      f(ci, &line[start..i]);
      ci += 1;
      start = i + 1;
    }
  }
  f(ci, &line[start..]);
}

fn split_line_fields<'a>(line: &'a [u8], delim: u8) -> Vec<&'a [u8]> {
  let mut out = Vec::new();
  let mut start = 0usize;
  for (i, &b) in line.iter().enumerate() {
    if b == delim {
      out.push(&line[start..i]);
      start = i + 1;
    }
  }
  out.push(&line[start..]);
  out
}

fn line_starts(bytes: &[u8]) -> Vec<usize> {
  let mut starts = vec![0usize];
  let mut i = 0usize;
  while i < bytes.len() {
    if bytes[i] == b'\n' {
      starts.push(i + 1);
    } else if bytes[i] == b'\r' {
      if i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
        starts.push(i + 2);
        i += 1;
      } else {
        starts.push(i + 1);
      }
    }
    i += 1;
  }
  if let Some(&last) = starts.last() {
    if last >= bytes.len() {
      starts.pop();
    }
  }
  starts
}

fn line_slice<'a>(bytes: &'a [u8], start: usize) -> &'a [u8] {
  let mut end = start;
  while end < bytes.len() && bytes[end] != b'\n' && bytes[end] != b'\r' {
    end += 1;
  }
  &bytes[start..end]
}

fn parse_header_names(line: &[u8], delim: u8) -> Vec<String> {
  split_line_fields(line, delim)
    .into_iter()
    .map(|f| String::from_utf8_lossy(f).into_owned())
    .collect()
}

/// Deduplicate headers like pandas: a, a.1, a.2 …
fn dedupe_headers(headers: Vec<String>) -> Vec<String> {
  let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
  headers
    .into_iter()
    .map(|h| {
      let k = seen.get(&h).copied();
      if k.is_none() {
        seen.insert(h.clone(), 0);
        h
      } else {
        let mut n = k.unwrap() + 1;
        loop {
          let cand = format!("{h}.{n}");
          if !seen.contains_key(&cand) {
            seen.insert(h.clone(), n);
            seen.insert(cand.clone(), 0);
            return cand;
          }
          n += 1;
        }
      }
    })
    .collect()
}

/// Fast path: unquoted CSV bytes → columnar table (Rayon over row chunks). Matches the JS fused
/// reader for default locale (`,` / `.`, no custom null/bool tokens). Returns `null` columns empty
/// on empty input.
#[napi]
pub fn parse_csv_unquoted(bytes: Buffer, delimiter: u32, has_header: bool) -> Result<NativeCsvTable> {
  let bytes = bytes.as_ref();
  if bytes.is_empty() {
    return Ok(NativeCsvTable {
      num_rows: 0,
      columns: vec![],
    });
  }
  let delim = delimiter as u8;
  if delim == 0 {
    return Err(Error::from_reason("delimiter must be a single byte"));
  }
  if bytes.contains(&b'"') {
    return Err(Error::from_reason("quoted CSV: use JS path"));
  }

  let starts = line_starts(bytes);
  if starts.is_empty() {
    return Ok(NativeCsvTable {
      num_rows: 0,
      columns: vec![],
    });
  }

  let mut data_from = 0usize;
  let headers = if has_header {
    let hline = line_slice(bytes, starts[0]);
    data_from = 1;
    dedupe_headers(parse_header_names(hline, delim))
  } else {
    let width = split_line_fields(line_slice(bytes, starts[0]), delim).len();
    (0..width).map(|i| format!("column_{i}")).collect()
  };
  let ncols = headers.len();
  if ncols == 0 {
    return Ok(NativeCsvTable {
      num_rows: 0,
      columns: vec![],
    });
  }

  let data_starts: Vec<usize> = starts.into_iter().skip(data_from).collect();
  let data_starts: Vec<usize> = data_starts
    .into_iter()
    .filter(|&s| {
      let line = line_slice(bytes, s);
      line.iter().any(|&b| b > b' ')
    })
    .collect();
  let n = data_starts.len();
  if n == 0 {
    return Ok(NativeCsvTable {
      num_rows: 0,
      columns: headers
        .into_iter()
        .map(|name| NativeCsvColumn {
          name,
          dtype: "f64".into(),
          null_bitmap: None,
          f64_data: Some(Float64Array::new(vec![])),
          i32_data: None,
          bool_data: None,
          cat_codes: None,
          dictionary: None,
          utf8_data: None,
        })
        .collect(),
    });
  }

  let chunk = ((n / rayon::current_num_threads().max(1)) + 1).max(4096);
  let parts: Vec<Vec<ColAcc>> = data_starts
    .par_chunks(chunk)
    .map(|chunk_starts| {
      let cap = chunk_starts.len();
      let mut cols: Vec<ColAcc> = headers
        .iter()
        .map(|h| ColAcc::with_capacity(h.clone(), cap))
        .collect();
      for &s in chunk_starts {
        let line = line_slice(bytes, s);
        let mut seen = 0usize;
        for_each_field(line, delim, |ci, raw| {
          if ci < ncols {
            push_parsed(&mut cols[ci], parse_field(raw));
            seen = ci + 1;
          }
        });
        for ci in seen..ncols {
          cols[ci].push_null();
        }
      }
      cols
    })
    .collect();

  let mut merged: Vec<ColAcc> = headers
    .iter()
    .map(|h| ColAcc::with_capacity(h.clone(), n))
    .collect();
  for part in parts {
    for (ci, col) in part.into_iter().enumerate() {
      merged[ci].append_merge(col);
    }
  }

  let columns: Vec<NativeCsvColumn> = merged.into_iter().map(|c| c.finish()).collect();
  Ok(NativeCsvTable {
    num_rows: n as u32,
    columns,
  })
}

#[inline]
fn bit_valid(bm: Option<&[u8]>, i: usize) -> bool {
  match bm {
    None => true,
    Some(b) => (b[i >> 3] >> (i & 7)) & 1 != 0,
  }
}

/// Write an unquoted CSV to `path`. Caller must ensure no cell needs quoting (no `,`/`"`/CRLF in text).
#[napi]
pub fn write_csv_unquoted(path: String, num_rows: u32, columns: Vec<NativeCsvWriteColumn>) -> Result<()> {
  let n = num_rows as usize;
  let ncols = columns.len();
  // Extract Sync-safe borrowed views (napi TypedArrays are not Sync).
  struct ColView<'a> {
    name: &'a str,
    dtype: &'a str,
    null_bitmap: Option<&'a [u8]>,
    f64_data: Option<&'a [f64]>,
    i32_data: Option<&'a [i32]>,
    bool_data: Option<&'a [u8]>,
    cat_codes: Option<&'a [u32]>,
    dictionary: Option<&'a [String]>,
    utf8_data: Option<&'a [String]>,
  }
  let views: Vec<ColView> = columns
    .iter()
    .map(|c| ColView {
      name: c.name.as_str(),
      dtype: c.dtype.as_str(),
      null_bitmap: c.null_bitmap.as_ref().map(|b| b.as_ref()),
      f64_data: c.f64_data.as_ref().map(|b| b.as_ref()),
      i32_data: c.i32_data.as_ref().map(|b| b.as_ref()),
      bool_data: c.bool_data.as_ref().map(|b| b.as_ref()),
      cat_codes: c.cat_codes.as_ref().map(|b| b.as_ref()),
      dictionary: c.dictionary.as_deref(),
      utf8_data: c.utf8_data.as_deref(),
    })
    .collect();

  let write_cell = |out: &mut Vec<u8>, col: &ColView, row: usize| {
    if !bit_valid(col.null_bitmap, row) {
      return;
    }
    match col.dtype {
      "i32" => {
        if let Some(data) = col.i32_data {
          let mut buf = itoa::Buffer::new();
          out.extend_from_slice(buf.format(data[row]).as_bytes());
        }
      }
      "f64" | "f32" | "datetime" => {
        if let Some(data) = col.f64_data {
          let v = data[row];
          if v.fract() == 0.0 && v.is_finite() && v.abs() <= (1u64 << 53) as f64 {
            let mut buf = itoa::Buffer::new();
            out.extend_from_slice(buf.format(v as i64).as_bytes());
          } else {
            let mut buf = ryu::Buffer::new();
            out.extend_from_slice(buf.format(v).as_bytes());
          }
        } else if let Some(data) = col.i32_data {
          let mut buf = itoa::Buffer::new();
          out.extend_from_slice(buf.format(data[row]).as_bytes());
        }
      }
      "bool" => {
        if let Some(data) = col.bool_data {
          out.extend_from_slice(if data[row] != 0 { b"true" } else { b"false" });
        }
      }
      "category" => {
        if let (Some(codes), Some(dict)) = (col.cat_codes, col.dictionary) {
          let code = codes[row] as usize;
          if let Some(s) = dict.get(code) {
            out.extend_from_slice(s.as_bytes());
          }
        }
      }
      "utf8" => {
        if let Some(data) = col.utf8_data {
          if let Some(s) = data.get(row) {
            out.extend_from_slice(s.as_bytes());
          }
        }
      }
      _ => {
        if let Some(data) = col.f64_data {
          let mut buf = ryu::Buffer::new();
          out.extend_from_slice(buf.format(data[row]).as_bytes());
        } else if let Some(data) = col.i32_data {
          let mut buf = itoa::Buffer::new();
          out.extend_from_slice(buf.format(data[row]).as_bytes());
        }
      }
    }
  };

  let chunk = ((n / rayon::current_num_threads().max(1)) + 1).max(4096);
  let parts: Vec<Vec<u8>> = (0..n)
    .into_par_iter()
    .chunks(chunk)
    .map(|rows| {
      let mut buf = Vec::with_capacity(rows.len().saturating_mul(ncols.max(1).saturating_mul(24)));
      for &row in rows.iter() {
        for (ci, col) in views.iter().enumerate() {
          if ci > 0 {
            buf.push(b',');
          }
          write_cell(&mut buf, col, row);
        }
        buf.push(b'\n');
      }
      buf
    })
    .collect();

  use std::io::Write;
  let mut file = std::io::BufWriter::with_capacity(
    1 << 20,
    std::fs::File::create(&path).map_err(|e| Error::from_reason(format!("writeCsv: {e}")))?,
  );
  for (ci, col) in views.iter().enumerate() {
    if ci > 0 {
      file.write_all(b",").map_err(|e| Error::from_reason(format!("writeCsv: {e}")))?;
    }
    file
      .write_all(col.name.as_bytes())
      .map_err(|e| Error::from_reason(format!("writeCsv: {e}")))?;
  }
  if n > 0 {
    file.write_all(b"\n").map_err(|e| Error::from_reason(format!("writeCsv: {e}")))?;
  }
  let last = parts.len().saturating_sub(1);
  for (i, part) in parts.into_iter().enumerate() {
    let bytes = if i == last && part.last() == Some(&b'\n') {
      &part[..part.len() - 1]
    } else {
      part.as_slice()
    };
    file
      .write_all(bytes)
      .map_err(|e| Error::from_reason(format!("writeCsv: {e}")))?;
  }
  file
    .flush()
    .map_err(|e| Error::from_reason(format!("writeCsv: {e}")))?;
  Ok(())
}
