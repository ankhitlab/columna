use wasm_bindgen::prelude::*;
use js_sys::{Float64Array, Int32Array, Uint32Array, Uint8Array};

fn is_valid(bitmap: &Uint8Array, index: usize) -> bool {
    if bitmap.length() == 0 {
        return true;
    }
    let byte = bitmap.get_index((index >> 3) as u32);
    (byte & (1u8 << (index & 7))) != 0
}

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

#[wasm_bindgen]
pub fn filter_mask(
    values: &Float64Array,
    op: u32,
    literal: f64,
    null_bitmap: &Uint8Array,
) -> Uint8Array {
    let n = values.length() as usize;
    let mut mask = vec![0u8; n];
    for i in 0..n {
        if !is_valid(null_bitmap, i) {
            continue;
        }
        let v = values.get_index(i as u32);
        if cmp_f64(op, v, literal) {
            mask[i] = 1;
        }
    }
    Uint8Array::from(mask.as_slice())
}

#[wasm_bindgen]
pub fn filter_mask_and2(
    a: &Float64Array,
    b: &Float64Array,
    op_a: u32,
    lit_a: f64,
    op_b: u32,
    lit_b: f64,
) -> Uint32Array {
    let n = a.length() as usize;
    let mut tmp: Vec<u32> = Vec::with_capacity(n / 2);
    for i in 0..n {
        let va = a.get_index(i as u32);
        let vb = b.get_index(i as u32);
        if cmp_f64(op_a, va, lit_a) && cmp_f64(op_b, vb, lit_b) {
            tmp.push(i as u32);
        }
    }
    Uint32Array::from(tmp.as_slice())
}

/// Typed dual filter: Int32 column AND Float64 column (bench: age + salary).
/// Bulk-copies into Rust vectors once, then runs a tight loop (avoids per-index JS glue).
#[wasm_bindgen]
pub fn filter_and2_i32_f64(
    a: &Int32Array,
    b: &Float64Array,
    op_a: u32,
    lit_a: f64,
    op_b: u32,
    lit_b: f64,
) -> Uint32Array {
    let n = a.length() as usize;
    let mut av = vec![0i32; n];
    let mut bv = vec![0f64; n];
    a.copy_to(&mut av);
    b.copy_to(&mut bv);
    let lit_ai = lit_a as i32;
    let mut tmp: Vec<u32> = Vec::with_capacity(n / 2);
    for i in 0..n {
        if cmp_i32(op_a, av[i], lit_ai) && cmp_f64(op_b, bv[i], lit_b) {
            tmp.push(i as u32);
        }
    }
    Uint32Array::from(tmp.as_slice())
}

/// Typed dual filter: two Int32 columns.
#[wasm_bindgen]
pub fn filter_and2_i32_i32(
    a: &Int32Array,
    b: &Int32Array,
    op_a: u32,
    lit_a: f64,
    op_b: u32,
    lit_b: f64,
) -> Uint32Array {
    let n = a.length() as usize;
    let mut av = vec![0i32; n];
    let mut bv = vec![0i32; n];
    a.copy_to(&mut av);
    b.copy_to(&mut bv);
    let lit_ai = lit_a as i32;
    let lit_bi = lit_b as i32;
    let mut tmp: Vec<u32> = Vec::with_capacity(n / 2);
    for i in 0..n {
        if cmp_i32(op_a, av[i], lit_ai) && cmp_i32(op_b, bv[i], lit_bi) {
            tmp.push(i as u32);
        }
    }
    Uint32Array::from(tmp.as_slice())
}

#[wasm_bindgen]
pub fn compact_indices(mask: &Uint8Array) -> Uint32Array {
    let mut out: Vec<u32> = Vec::new();
    for i in 0..mask.length() {
        if mask.get_index(i) != 0 {
            out.push(i);
        }
    }
    Uint32Array::from(out.as_slice())
}

#[wasm_bindgen]
pub fn sort_indices(
    values: &Float64Array,
    descending: bool,
    null_bitmap: &Uint8Array,
) -> Uint32Array {
    let n = values.length() as usize;
    let mut idx: Vec<u32> = (0..n as u32).collect();
    idx.sort_by(|&a, &b| {
        let a_null = !is_valid(null_bitmap, a as usize);
        let b_null = !is_valid(null_bitmap, b as usize);
        match (a_null, b_null) {
            (true, true) => a.cmp(&a),
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => {
                let av = values.get_index(a);
                let bv = values.get_index(b);
                let ord = av.total_cmp(&bv);
                if descending {
                    ord.reverse()
                } else {
                    ord
                }
            }
        }
    });
    Uint32Array::from(idx.as_slice())
}

#[wasm_bindgen]
pub struct GroupByResult {
    keys: Uint32Array,
    values: Float64Array,
}

#[wasm_bindgen]
impl GroupByResult {
    #[wasm_bindgen(getter)]
    pub fn keys(&self) -> Uint32Array {
        self.keys.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Float64Array {
        self.values.clone()
    }
}

#[wasm_bindgen]
pub fn hash_group_by(
    key_hashes: &Uint32Array,
    values: &Float64Array,
    agg_op: u32,
    null_bitmap: &Uint8Array,
) -> GroupByResult {
    use std::collections::BTreeMap;
    let n = key_hashes.length() as usize;
    let mut map: BTreeMap<u32, (u32, f64, f64, f64, f64)> = BTreeMap::new();
    for i in 0..n {
        if !is_valid(null_bitmap, i) {
            continue;
        }
        let k = key_hashes.get_index(i as u32);
        let v = values.get_index(i as u32);
        let entry = map.entry(k).or_insert((0, 0.0, v, v, v));
        entry.0 += 1;
        entry.1 += v;
        entry.2 = entry.2.min(v);
        entry.3 = entry.3.max(v);
        entry.4 = v;
    }
    let mut keys = Vec::with_capacity(map.len());
    let mut out = Vec::with_capacity(map.len());
    for (k, (count, sum, min, max, last)) in map {
        keys.push(k);
        let val = match agg_op {
            0 => sum,
            1 => sum / count as f64,
            2 => min,
            3 => max,
            4 => count as f64,
            _ => last,
        };
        out.push(val);
    }
    GroupByResult {
        keys: Uint32Array::from(keys.as_slice()),
        values: Float64Array::from(out.as_slice()),
    }
}
