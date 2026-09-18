"""Writes tests/fixtures/polars-ipc.arrows (stream) and polars-ipc.arrow (file) with Polars — a Rust Arrow
writer, independent of apache-arrow JS: Utf8View strings, Categorical / Enum dictionaries, Datetime(us),
Date32, Int8, UInt64, nullable booleans. Run: py -3 tests/fixtures/make_polars_ipc_fixture.py"""
import datetime, pathlib
import polars as pl

df = pl.DataFrame({
    'f': [1.5, None, -2.25, 3.0],
    'i8': pl.Series([1, -2, None, 4], dtype=pl.Int8),
    'u64': pl.Series([0, 2**53 - 1, None, 7], dtype=pl.UInt64),
    's': ['alpha', '', 'a very long string that does not fit in twelve bytes 🚀', None],
    'c': pl.Series(['x', 'y', None, 'x'], dtype=pl.Categorical),
    'e': pl.Series(['lo', 'hi', 'lo', None], dtype=pl.Enum(['lo', 'hi'])),
    'd': [datetime.datetime(2024, 1, 1), None, datetime.datetime(1969, 12, 31, 23, 59, 59, 999000), datetime.datetime(2100, 6, 15, 12)],
    'day': [datetime.date(2024, 1, 1), None, datetime.date(1970, 1, 2), datetime.date(1969, 12, 31)],
    'b': [True, None, False, True],
})
here = pathlib.Path(__file__).parent
df.write_ipc_stream(here / 'polars-ipc.arrows')
df.write_ipc(here / 'polars-ipc.arrow')
print('polars', pl.__version__, df.schema)
