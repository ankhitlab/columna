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
