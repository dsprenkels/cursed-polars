"""
AES-256 implemented entirely in Polars LazyFrame operations.

Each function takes a LazyFrame and returns a LazyFrame. The key schedule
expands into individual columns (ks_0 … ks_239), and the AES state lives
in columns (s_0 … s_15) that are overwritten at each round step via
with_columns.  The resulting query plans are enormous — that's the point.
"""

import polars as pl

# fmt: off
SBOX = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
]

INV_SBOX = [
    0x52,0x09,0x6a,0xd5,0x30,0x36,0xa5,0x38,0xbf,0x40,0xa3,0x9e,0x81,0xf3,0xd7,0xfb,
    0x7c,0xe3,0x39,0x82,0x9b,0x2f,0xff,0x87,0x34,0x8e,0x43,0x44,0xc4,0xde,0xe9,0xcb,
    0x54,0x7b,0x94,0x32,0xa6,0xc2,0x23,0x3d,0xee,0x4c,0x95,0x0b,0x42,0xfa,0xc3,0x4e,
    0x08,0x2e,0xa1,0x66,0x28,0xd9,0x24,0xb2,0x76,0x5b,0xa2,0x49,0x6d,0x8b,0xd1,0x25,
    0x72,0xf8,0xf6,0x64,0x86,0x68,0x98,0x16,0xd4,0xa4,0x5c,0xcc,0x5d,0x65,0xb6,0x92,
    0x6c,0x70,0x48,0x50,0xfd,0xed,0xb9,0xda,0x5e,0x15,0x46,0x57,0xa7,0x8d,0x9d,0x84,
    0x90,0xd8,0xab,0x00,0x8c,0xbc,0xd3,0x0a,0xf7,0xe4,0x58,0x05,0xb8,0xb3,0x45,0x06,
    0xd0,0x2c,0x1e,0x8f,0xca,0x3f,0x0f,0x02,0xc1,0xaf,0xbd,0x03,0x01,0x13,0x8a,0x6b,
    0x3a,0x91,0x11,0x41,0x4f,0x67,0xdc,0xea,0x97,0xf2,0xcf,0xce,0xf0,0xb4,0xe6,0x73,
    0x96,0xac,0x74,0x22,0xe7,0xad,0x35,0x85,0xe2,0xf9,0x37,0xe8,0x1c,0x75,0xdf,0x6e,
    0x47,0xf1,0x1a,0x71,0x1d,0x29,0xc5,0x89,0x6f,0xb7,0x62,0x0e,0xaa,0x18,0xbe,0x1b,
    0xfc,0x56,0x3e,0x4b,0xc6,0xd2,0x79,0x20,0x9a,0xdb,0xc0,0xfe,0x78,0xcd,0x5a,0xf4,
    0x1f,0xdd,0xa8,0x33,0x88,0x07,0xc7,0x31,0xb1,0x12,0x10,0x59,0x27,0x80,0xec,0x5f,
    0x60,0x51,0x7f,0xa9,0x19,0xb5,0x4a,0x0d,0x2d,0xe5,0x7a,0x9f,0x93,0xc9,0x9c,0xef,
    0xa0,0xe0,0x3b,0x4d,0xae,0x2a,0xf5,0xb0,0xc8,0xeb,0xbb,0x3c,0x83,0x53,0x99,0x61,
    0x17,0x2b,0x04,0x7e,0xba,0x77,0xd6,0x26,0xe1,0x69,0x14,0x63,0x55,0x21,0x0c,0x7d,
]

RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]
# fmt: on


# ---------------------------------------------------------------------------
# Byte-level expression helpers
# ---------------------------------------------------------------------------

def _sbox(b: pl.Expr) -> pl.Expr:
    return pl.lit(SBOX).list.get(b.cast(pl.Int64)).cast(pl.UInt8)


def _inv_sbox(b: pl.Expr) -> pl.Expr:
    return pl.lit(INV_SBOX).list.get(b.cast(pl.Int64)).cast(pl.UInt8)


def _xtime(b: pl.Expr) -> pl.Expr:
    """Multiply by 2 in GF(2^8)."""
    wide = b.cast(pl.UInt16)
    return (((wide * 2) & 0xFF) ^ pl.when(wide & 0x80 > 0).then(pl.lit(0x1B, dtype=pl.UInt16)).otherwise(pl.lit(0, dtype=pl.UInt16))).cast(pl.UInt8)


def _gf_mul(a: pl.Expr, c: int) -> pl.Expr:
    """Multiply expression *a* by compile-time constant *c* in GF(2^8)."""
    if c == 1:  return a.cast(pl.UInt8)
    if c == 2:  return _xtime(a)
    if c == 3:  return _xtime(a) ^ a.cast(pl.UInt8)
    t2 = _xtime(a); t4 = _xtime(t2); t8 = _xtime(t4)
    if c == 9:  return t8 ^ a.cast(pl.UInt8)
    if c == 11: return t8 ^ t2 ^ a.cast(pl.UInt8)
    if c == 13: return t8 ^ t4 ^ a.cast(pl.UInt8)
    if c == 14: return t8 ^ t4 ^ t2
    raise ValueError(c)


# State layout (column-major):
#   0  4  8 12
#   1  5  9 13
#   2  6 10 14
#   3  7 11 15
SHIFT_ROWS     = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11]
INV_SHIFT_ROWS = [0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3]

def _ks(i: int) -> str: return f"ks_{i}"
def _s(i: int) -> str:  return f"s_{i}"


# ---------------------------------------------------------------------------
# Key schedule  (LazyFrame -> LazyFrame)
# ---------------------------------------------------------------------------

def aes256_key_schedule(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Expects a ``key`` column (Binary, 32 bytes).
    Adds columns ``ks_0`` ... ``ks_239`` (UInt8) -- the full AES-256 key schedule.
    """
    key = pl.col("key").cast(pl.List(pl.UInt8))
    lf = lf.with_columns([
        key.list.get(i).cast(pl.UInt8).alias(_ks(i)) for i in range(32)
    ])

    Nk = 8
    for wi in range(Nk, 60):
        prev = [pl.col(_ks((wi - 1) * 4 + j)) for j in range(4)]
        back = [pl.col(_ks((wi - Nk) * 4 + j)) for j in range(4)]

        if wi % Nk == 0:
            rot = [prev[1], prev[2], prev[3], prev[0]]
            sub = [_sbox(b) for b in rot]
            rcon = RCON[wi // Nk - 1]
            new = [
                (back[0] ^ sub[0] ^ pl.lit(rcon, dtype=pl.UInt8)).cast(pl.UInt8),
                (back[1] ^ sub[1]).cast(pl.UInt8),
                (back[2] ^ sub[2]).cast(pl.UInt8),
                (back[3] ^ sub[3]).cast(pl.UInt8),
            ]
        elif wi % Nk == 4:
            sub = [_sbox(b) for b in prev]
            new = [(back[j] ^ sub[j]).cast(pl.UInt8) for j in range(4)]
        else:
            new = [(back[j] ^ prev[j]).cast(pl.UInt8) for j in range(4)]

        lf = lf.with_columns([new[j].alias(_ks(wi * 4 + j)) for j in range(4)])

    return lf


# ---------------------------------------------------------------------------
# Round-step helpers  (LazyFrame -> LazyFrame)
# ---------------------------------------------------------------------------

def _add_round_key(lf: pl.LazyFrame, rnd: int) -> pl.LazyFrame:
    off = rnd * 16
    return lf.with_columns([
        (pl.col(_s(i)) ^ pl.col(_ks(off + i))).cast(pl.UInt8).alias(_s(i))
        for i in range(16)
    ])


def _sub_bytes_step(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns([_sbox(pl.col(_s(i))).alias(_s(i)) for i in range(16)])


def _inv_sub_bytes_step(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns([_inv_sbox(pl.col(_s(i))).alias(_s(i)) for i in range(16)])


def _shift_rows_step(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns([
        pl.col(_s(SHIFT_ROWS[i])).alias(_s(i)) for i in range(16)
    ])


def _inv_shift_rows_step(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns([
        pl.col(_s(INV_SHIFT_ROWS[i])).alias(_s(i)) for i in range(16)
    ])


def _mix_columns_step(lf: pl.LazyFrame) -> pl.LazyFrame:
    exprs: list[pl.Expr] = []
    for c in range(4):
        s = [pl.col(_s(c * 4 + r)) for r in range(4)]
        exprs.append((_gf_mul(s[0], 2) ^ _gf_mul(s[1], 3) ^ s[2].cast(pl.UInt8) ^ s[3].cast(pl.UInt8)).cast(pl.UInt8).alias(_s(c * 4 + 0)))
        exprs.append((s[0].cast(pl.UInt8) ^ _gf_mul(s[1], 2) ^ _gf_mul(s[2], 3) ^ s[3].cast(pl.UInt8)).cast(pl.UInt8).alias(_s(c * 4 + 1)))
        exprs.append((s[0].cast(pl.UInt8) ^ s[1].cast(pl.UInt8) ^ _gf_mul(s[2], 2) ^ _gf_mul(s[3], 3)).cast(pl.UInt8).alias(_s(c * 4 + 2)))
        exprs.append((_gf_mul(s[0], 3) ^ s[1].cast(pl.UInt8) ^ s[2].cast(pl.UInt8) ^ _gf_mul(s[3], 2)).cast(pl.UInt8).alias(_s(c * 4 + 3)))
    return lf.with_columns(exprs)


def _inv_mix_columns_step(lf: pl.LazyFrame) -> pl.LazyFrame:
    exprs: list[pl.Expr] = []
    for c in range(4):
        s = [pl.col(_s(c * 4 + r)) for r in range(4)]
        exprs.append((_gf_mul(s[0], 14) ^ _gf_mul(s[1], 11) ^ _gf_mul(s[2], 13) ^ _gf_mul(s[3], 9)).cast(pl.UInt8).alias(_s(c * 4 + 0)))
        exprs.append((_gf_mul(s[0], 9)  ^ _gf_mul(s[1], 14) ^ _gf_mul(s[2], 11) ^ _gf_mul(s[3], 13)).cast(pl.UInt8).alias(_s(c * 4 + 1)))
        exprs.append((_gf_mul(s[0], 13) ^ _gf_mul(s[1], 9)  ^ _gf_mul(s[2], 14) ^ _gf_mul(s[3], 11)).cast(pl.UInt8).alias(_s(c * 4 + 2)))
        exprs.append((_gf_mul(s[0], 11) ^ _gf_mul(s[1], 13) ^ _gf_mul(s[2], 9)  ^ _gf_mul(s[3], 14)).cast(pl.UInt8).alias(_s(c * 4 + 3)))
    return lf.with_columns(exprs)


# ---------------------------------------------------------------------------
# Core AES-256 block cipher (no IV, no key expansion)
# ---------------------------------------------------------------------------

def _aes256_ecb_encrypt_core(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Internal: run all 14+1 AES-256 rounds on s_0..s_15 using ks_0..ks_239.
    Caller must set up s_* state and ks_* schedule before calling.
    """
    lf = _add_round_key(lf, 0)
    for rnd in range(1, 14):
        lf = _sub_bytes_step(lf)
        lf = _shift_rows_step(lf)
        lf = _mix_columns_step(lf)
        lf = _add_round_key(lf, rnd)
    lf = _sub_bytes_step(lf)
    lf = _shift_rows_step(lf)
    lf = _add_round_key(lf, 14)
    return lf


def _aes256_ecb_decrypt_core(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Internal: run all inverse AES-256 rounds on s_0..s_15 using ks_0..ks_239.
    """
    lf = _add_round_key(lf, 14)
    lf = _inv_shift_rows_step(lf)
    lf = _inv_sub_bytes_step(lf)
    for rnd in range(13, 0, -1):
        lf = _add_round_key(lf, rnd)
        lf = _inv_mix_columns_step(lf)
        lf = _inv_shift_rows_step(lf)
        lf = _inv_sub_bytes_step(lf)
    lf = _add_round_key(lf, 0)
    return lf


# ---------------------------------------------------------------------------
# CBC single-block encrypt/decrypt -- with pre-computed schedule
# ---------------------------------------------------------------------------

def aes256_encrypt_block_with_schedule(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    AES-256-CBC encrypt one block using an already-computed key schedule.

    Expects: ``plaintext`` (Binary 16), ``iv`` (Binary 16), ``ks_0``..``ks_239``.
    Adds: ``ciphertext`` (Binary 16). Drops all ks_* and s_* intermediates.
    """
    pt = pl.col("plaintext").cast(pl.List(pl.UInt8))
    iv = pl.col("iv").cast(pl.List(pl.UInt8))
    lf = lf.with_columns([
        (pt.list.get(i).cast(pl.UInt8) ^ iv.list.get(i).cast(pl.UInt8)).cast(pl.UInt8).alias(_s(i))
        for i in range(16)
    ])
    lf = _aes256_ecb_encrypt_core(lf)
    lf = lf.with_columns(
        pl.concat_list([pl.col(_s(i)) for i in range(16)]).cast(pl.Binary).alias("ciphertext")
    )
    lf = lf.drop([_s(i) for i in range(16)] + [_ks(i) for i in range(240)])
    return lf


def aes256_decrypt_block_with_schedule(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    AES-256-CBC decrypt one block using an already-computed key schedule.

    Expects: ``ciphertext`` (Binary 16), ``iv`` (Binary 16), ``ks_0``..``ks_239``.
    Adds: ``decrypted`` (Binary 16). Drops all ks_* and s_* intermediates.
    """
    ct = pl.col("ciphertext").cast(pl.List(pl.UInt8))
    lf = lf.with_columns([
        ct.list.get(i).cast(pl.UInt8).alias(_s(i)) for i in range(16)
    ])
    lf = _aes256_ecb_decrypt_core(lf)
    iv = pl.col("iv").cast(pl.List(pl.UInt8))
    lf = lf.with_columns([
        (pl.col(_s(i)) ^ iv.list.get(i).cast(pl.UInt8)).cast(pl.UInt8).alias(_s(i))
        for i in range(16)
    ])
    lf = lf.with_columns(
        pl.concat_list([pl.col(_s(i)) for i in range(16)]).cast(pl.Binary).alias("decrypted")
    )
    lf = lf.drop([_s(i) for i in range(16)] + [_ks(i) for i in range(240)])
    return lf


# ---------------------------------------------------------------------------
# CBC single-block encrypt/decrypt -- full (includes key schedule)
# ---------------------------------------------------------------------------

def aes256_encrypt_block(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    AES-256-CBC encrypt one block.

    Expects: ``key`` (Binary 32), ``plaintext`` (Binary 16), ``iv`` (Binary 16).
    Adds: ``ciphertext`` (Binary 16).
    """
    return aes256_encrypt_block_with_schedule(aes256_key_schedule(lf))


def aes256_decrypt_block(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    AES-256-CBC decrypt one block.

    Expects: ``key`` (Binary 32), ``ciphertext`` (Binary 16), ``iv`` (Binary 16).
    Adds: ``decrypted`` (Binary 16).
    """
    return aes256_decrypt_block_with_schedule(aes256_key_schedule(lf))


# ---------------------------------------------------------------------------
# CTR mode  (LazyFrame -> LazyFrame)
# ---------------------------------------------------------------------------

def aes256_ctr_encrypt(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    AES-256-CTR encryption (and decryption -- CTR is symmetric).

    Each row is one 16-byte block. The row index (from ``with_row_index``) is
    used as the 32-bit big-endian block counter appended to the 12-byte nonce,
    matching the NIST/GCM counter block layout:

        counter_block = nonce[0..11] || uint32_be(row_index)

    Expects: ``key`` (Binary 32), ``nonce`` (Binary 12), ``plaintext`` (Binary 16).
    Adds: ``ciphertext`` (Binary 16).
    """
    lf = lf.with_row_index("_ctr_idx")
    lf = aes256_key_schedule(lf)

    # Build 16-byte counter block in s_0..s_15:
    #   s_0..s_11 = nonce bytes
    #   s_12..s_15 = row_index as big-endian uint32
    nonce = pl.col("nonce").cast(pl.List(pl.UInt8))
    idx   = pl.col("_ctr_idx").cast(pl.UInt32)
    lf = lf.with_columns(
        [nonce.list.get(i).cast(pl.UInt8).alias(_s(i)) for i in range(12)] +
        [(idx // 0x01000000       ).cast(pl.UInt8).alias(_s(12)),
         ((idx // 0x00010000) % 256).cast(pl.UInt8).alias(_s(13)),
         ((idx // 0x00000100) % 256).cast(pl.UInt8).alias(_s(14)),
         ( idx                % 256).cast(pl.UInt8).alias(_s(15))]
    )

    # AES-encrypt the counter block to produce the keystream block
    lf = _aes256_ecb_encrypt_core(lf)

    # XOR keystream with plaintext
    pt = pl.col("plaintext").cast(pl.List(pl.UInt8))
    lf = lf.with_columns([
        (pl.col(_s(i)) ^ pt.list.get(i).cast(pl.UInt8)).cast(pl.UInt8).alias(_s(i))
        for i in range(16)
    ])

    lf = lf.with_columns(
        pl.concat_list([pl.col(_s(i)) for i in range(16)]).cast(pl.Binary).alias("ciphertext")
    )
    lf = lf.drop([_s(i) for i in range(16)] + [_ks(i) for i in range(240)] + ["_ctr_idx"])
    return lf


# CTR decrypt is identical to CTR encrypt
aes256_ctr_decrypt = aes256_ctr_encrypt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_aes256_cbc_roundtrip():
    import os
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    key = os.urandom(32)
    iv  = os.urandom(16)
    pt  = os.urandom(16)

    enc = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    ref_ct = enc.update(pt) + enc.finalize()

    lf = pl.LazyFrame({"key": [key], "iv": [iv], "plaintext": [pt]})
    df = aes256_encrypt_block(lf).collect()
    ct = df["ciphertext"][0]
    assert ct == ref_ct, f"CBC encrypt mismatch: {ct.hex()} != {ref_ct.hex()}"

    lf2 = pl.LazyFrame({"key": [key], "iv": [iv], "ciphertext": [ct]})
    df2 = aes256_decrypt_block(lf2).collect()
    assert df2["decrypted"][0] == pt
    print("  CBC single-row roundtrip OK")


def test_aes256_cbc_multi_row():
    import os
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    n = 5
    keys = [os.urandom(32) for _ in range(n)]
    ivs  = [os.urandom(16) for _ in range(n)]
    pts  = [os.urandom(16) for _ in range(n)]

    ref_cts = [
        (lambda e: e.update(p) + e.finalize())(Cipher(algorithms.AES(k), modes.CBC(iv)).encryptor())
        for k, iv, p in zip(keys, ivs, pts)
    ]

    lf = pl.LazyFrame({"key": keys, "iv": ivs, "plaintext": pts})
    df = aes256_encrypt_block(lf).collect()
    for i in range(n):
        assert df["ciphertext"][i] == ref_cts[i], f"Row {i} CBC encrypt mismatch"

    lf2 = pl.LazyFrame({"key": keys, "iv": ivs, "ciphertext": df["ciphertext"].to_list()})
    df2 = aes256_decrypt_block(lf2).collect()
    for i in range(n):
        assert df2["decrypted"][i] == pts[i], f"Row {i} CBC decrypt mismatch"
    print(f"  CBC {n}-row element-wise roundtrip OK")


def test_aes256_ctr_roundtrip():
    import os
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    n = 8  # 8 blocks = 128 bytes
    key   = os.urandom(32)
    nonce = os.urandom(12)
    pts   = [os.urandom(16) for _ in range(n)]

    # cryptography CTR: initial counter block = nonce || uint32_be(0)
    initial_ctr_block = nonce + b"\x00\x00\x00\x00"
    enc = Cipher(algorithms.AES(key), modes.CTR(initial_ctr_block)).encryptor()
    ref_ct_flat = enc.update(b"".join(pts)) + enc.finalize()
    ref_cts = [ref_ct_flat[i * 16:(i + 1) * 16] for i in range(n)]

    lf = pl.LazyFrame({
        "key":       [key] * n,
        "nonce":     [nonce] * n,
        "plaintext": pts,
    })
    df = aes256_ctr_encrypt(lf).collect()
    for i in range(n):
        assert df["ciphertext"][i] == ref_cts[i], f"CTR block {i} mismatch: {df['ciphertext'][i].hex()} != {ref_cts[i].hex()}"

    # Decrypt is symmetric
    lf2 = pl.LazyFrame({
        "key":       [key] * n,
        "nonce":     [nonce] * n,
        "plaintext": df["ciphertext"].to_list(),
    })
    df2 = aes256_ctr_decrypt(lf2).collect()
    for i in range(n):
        assert df2["ciphertext"][i] == pts[i], f"CTR decrypt block {i} mismatch"
    print(f"  CTR {n}-block roundtrip OK")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(n_blocks: int = 1000, engine: str = "streaming") -> None:
    """
    Measure CTR-mode throughput. Each row is one 16-byte AES block.
    Reports MB/s based on wall-clock time for collect().
    """
    import os, time

    key   = os.urandom(32)
    nonce = os.urandom(12)
    pts   = [os.urandom(16) for _ in range(n_blocks)]

    lf = pl.LazyFrame({
        "key":       [key] * n_blocks,
        "nonce":     [nonce] * n_blocks,
        "plaintext": pts,
    })

    t0 = time.perf_counter()
    lazy_result = aes256_ctr_encrypt(lf)
    lazy_result.collect(engine=engine)
    elapsed = time.perf_counter() - t0

    total_bytes = n_blocks * 16
    mb = total_bytes / (1024 * 1024)
    print(f"  {n_blocks:,} blocks  ({mb:.2f} MB)  in {elapsed:.3f} s  =>  {mb / elapsed:.2f} MB/s")


if __name__ == "__main__":
    print("=== Tests ===")
    test_aes256_cbc_roundtrip()
    test_aes256_cbc_multi_row()
    test_aes256_ctr_roundtrip()
    print("All tests passed!")
    print()
    print("=== Benchmark ===")
    for n in [1_000, 10_000, 100_000, 1_000_000, 10_000_000]:
        benchmark(n)
