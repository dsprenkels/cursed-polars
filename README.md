# Cursed polars recipes

## AES encryption

Who needs Parquet encryption and decryption if you can just encrypt/decrypt
your data directly in [Polars](https://pola.rs)? :D

This AES-256 implementation is completely implemented as a `LazyFrame` — no
eager evaluation, no UDFs, no plugins.  Every AES operation (S-box lookup, GF
multiplication, key schedule, round functions) is expressed as a chain of
`with_columns` calls that Polars compiles into one enormous query plan.

On my mediocre home pc:

```
       1,000 blocks  (  0.02 MB)  in 0.072 s  =>  0.21 MB/s
      10,000 blocks  (  0.15 MB)  in 0.063 s  =>  2.43 MB/s
     100,000 blocks  (  1.53 MB)  in 0.100 s  =>  15.29 MB/s
   1,000,000 blocks  ( 15.26 MB)  in 0.488 s  =>  31.25 MB/s
  10,000,000 blocks  (152.59 MB)  in 4.652 s  =>  32.80 MB/s
```
