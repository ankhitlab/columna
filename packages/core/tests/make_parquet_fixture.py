from pathlib import Path

try:
    import polars as pl
except ImportError:
    import pandas as pd

p = Path("packages/core/tests/fixtures")
p.mkdir(parents=True, exist_ok=True)
out = p / "sample.parquet"

try:
    import polars as pl

    pl.DataFrame(
        {"city": ["Berlin", "Paris"], "age": pl.Series([30, 41], dtype=pl.Int32)}
    ).write_parquet(out)
except ImportError:
    import pandas as pd

    pd.DataFrame({"city": ["Berlin", "Paris"], "age": pd.Series([30, 41], dtype="int32")}).to_parquet(out)

print("wrote", out, "size", out.stat().st_size)
