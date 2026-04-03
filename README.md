# Cursed polars recipes

## AES encryption

Who needs Parquet encryption and decryption if you can just encrypt/decrypt
your data directly in [Polars](https://pola.rs)? :D

This AES-256 implementation is completely implemented as a `LazyFrame` — no
eager evaluation, no UDFs, no plugins.  Every AES operation (S-box lookup, GF
multiplication, key schedule, round functions) is expressed as a chain of
`with_columns` calls that Polars compiles into one enormous query plan.

```
  1,000 blocks  (0.02 MB)  in 0.097 s  =>  0.16 MB/s
 10,000 blocks  (0.15 MB)  in 0.118 s  =>  1.29 MB/s
100,000 blocks  (1.53 MB)  in 0.290 s  =>  5.26 MB/s
```
