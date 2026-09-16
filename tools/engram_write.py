#!/usr/bin/env python3
"""Write a pack dir whose Engram tables are MXINT-4 with fp8-e4m3 block scales.

RECONSTRUCTED 2026-09-14 after a reboot wiped /tmp. Round-trip verified at rel_L2 0.102
against the fp8 reference. See BENCH-RESULTS.md "Engram MXINT-4 quantization".

<out> = hardlinks to every shard holding no Engram tensor, plus freshly written shards for
the two that do (Engram tensors quantized; every other tensor in those shards copied
verbatim), plus every top-level non-shard file (index/config/tokenizer - without these the
pack does not load at all). Tensor names and shard assignment are unchanged.

Layout per table: codes [N, cols/2] U8 (2 codes/byte, low nibble = even column) followed by
scales [N, cols/32] F8_E4M3. value = code_signed * (scale / prescale).
One read pass: codes stream out, scales go to a temp file appended after the code region.
"""
import argparse, json, os, shutil, struct, time
import torch

DT_SIZE = {"U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "BOOL": 1,
           "F16": 2, "BF16": 2, "I16": 2, "F32": 4, "I32": 4, "F64": 8, "I64": 8}

def nbytes(dt, shape):
    n = 1
    for d in shape:
        n *= d
    return n * DT_SIZE[dt]

def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return 8 + n, json.loads(fh.read(n))

def iter_rows(path, off, rows, cols, chunk):
    with open(path, "rb") as fh:
        for r0 in range(0, rows, chunk):
            r1 = min(r0 + chunk, rows)
            fh.seek(off + r0 * cols)
            yield r0, torch.frombuffer(bytearray(fh.read((r1 - r0) * cols)),
                                       dtype=torch.uint8).view(r1 - r0, cols)

def quant_bits(ref, block, prescale, bits):
    hi = (1 << (bits - 1)) - 1  # 7 at 4 bits, 3 at 3 bits
    """step quantizes the values; STORED is step*prescale so the scale sits in e4m3's normal
    range (below 2**-6 e4m3 goes subnormal and sheds mantissa bits). Consumers divide the
    stored scale by prescale. The prescale is NOT baked into the codes."""
    R, cols = ref.shape
    nb = cols // block
    b = ref.view(R, nb, block)
    amax = b.abs().amax(dim=2, keepdim=True)
    step = torch.where(amax > 0, amax / hi, torch.ones_like(amax))
    code = torch.round(b / step).clamp(-hi - 1, hi).to(torch.int16).view(R, cols)
    return code, (step * prescale).view(R, nb)

def pack_3bits(code):
    R, cols = code.shape
    c = (code.to(torch.int32) & 0x7).view(R, cols // 8, 8)
    shifts = torch.arange(8, dtype=torch.int32) * 3
    v = (c << shifts).sum(dim=2)
    b0 = (v & 0xFF).to(torch.uint8)
    b1 = ((v >> 8) & 0xFF).to(torch.uint8)
    b2 = ((v >> 16) & 0xFF).to(torch.uint8)
    return torch.stack([b0, b1, b2], dim=2).view(R, cols // 8 * 3)

def pack_nibbles(code):
    c = (code.to(torch.int32) & 0xF).to(torch.uint8)
    return (c[:, 0::2] | (c[:, 1::2] << 4)).to(torch.uint8)

ap = argparse.ArgumentParser()
ap.add_argument("--pack", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--layers", default="1,14")
ap.add_argument("--bits", type=int, default=4, choices=(3, 4))
ap.add_argument("--block", type=int, default=32)
ap.add_argument("--prescale", type=float, default=8.0)
ap.add_argument("--chunk-rows", type=int, default=1 << 16)
ap.add_argument("--limit-rows", type=int, default=0, help="debug: stop early")
a = ap.parse_args()

idx = json.load(open(f"{a.pack}/model.safetensors.index.json"))["weight_map"]
os.makedirs(a.out, exist_ok=True)
touched = set()

for L in (int(x) for x in a.layers.split(",")):
    wn, sn = f"layers.{L}.engram.embed.weight", f"layers.{L}.engram.embed.scale"
    shard = idx[wn]
    assert idx[sn] == shard
    touched.add(shard)
    data_off, hdr = read_header(f"{a.pack}/{shard}")
    N, cols = hdr[wn]["shape"]
    nb = cols // a.block
    src_nb = cols // 32          # source packs store ue8m0 scales per 32 columns
    assert hdr[sn]["shape"][1] == src_nb, (hdr[sn]["shape"], src_nb)
    wbytes, sbytes = (cols // 2 if a.bits == 4 else cols * 3 // 8), nb
    if a.limit_rows:
        N = min(N, a.limit_rows)
    others = [n for n in hdr if not n.startswith(f"layers.{L}.engram.embed.")]
    print(f"layer {L}: {N:,} x {cols} -> codes {wbytes} B/row + scales {sbytes} B/row "
          f"({(wbytes+sbytes)*N/1024**3:.1f} GiB)  others: {others}", flush=True)

    out_path = f"{a.out}/{shard}"
    stmp = out_path + ".stmp"
    order = others + [wn, sn]
    sizes, off = {}, 0
    for n in order:
        if n == wn:   sz = N * wbytes
        elif n == sn: sz = N * sbytes
        else:         sz = nbytes(hdr[n]["dtype"], hdr[n]["shape"])
        sizes[n] = (off, off + sz); off += sz
    blob = json.dumps({n: {"dtype": (hdr[n]["dtype"] if n in others else
                                    ("U8" if n == wn else "F8_E4M3")),
                           "shape": ([N, wbytes] if n == wn else
                                     [N, sbytes] if n == sn else list(hdr[n]["shape"])),
                           "data_offsets": list(sizes[n])} for n in order},
                      separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)
    s_off_rows = data_off + hdr[sn]["data_offsets"][0]
    s_off_w = data_off + hdr[wn]["data_offsets"][0]
    _thirty_two = 32

    t0 = time.time()
    with open(out_path, "wb") as out, open(stmp, "wb") as sf:
        out.write(struct.pack("<Q", len(blob))); out.write(blob)
        for n in others:
            src, end = data_off + hdr[n]["data_offsets"][0], data_off + hdr[n]["data_offsets"][1]
            assert end - src == nbytes(hdr[n]["dtype"], hdr[n]["shape"]), f"size mismatch {n}"
            with open(f"{a.pack}/{shard}", "rb") as g:
                g.seek(src)
                remaining = end - src
                while remaining > 0:
                    b = g.read(min(1 << 24, remaining))
                    if not b:
                        raise IOError("short read copying " + n)
                    out.write(b); remaining -= len(b)
        for r0, w in iter_rows(f"{a.pack}/{shard}", s_off_w, N, cols, a.chunk_rows):
            with open(f"{a.pack}/{shard}", "rb") as g2:
                g2.seek(s_off_rows + r0 * src_nb)
                sb = torch.frombuffer(bytearray(g2.read(w.shape[0] * src_nb)),
                                      dtype=torch.uint8).view(w.shape[0], src_nb)
            ref = (w.view(torch.float8_e4m3fn).to(torch.float32)
                   * sb.view(torch.float8_e8m0fnu).to(torch.float32).repeat_interleave(_thirty_two, dim=1))
            code, scale = quant_bits(ref, a.block, a.prescale, a.bits)
            out.write((pack_nibbles(code) if a.bits == 4 else pack_3bits(code)).numpy().tobytes())
            sf.write(scale.to(torch.float8_e4m3fn).view(torch.uint8).numpy().tobytes())
            if (r0 // a.chunk_rows) % 128 == 0:
                el = time.time() - t0; done = r0 + w.shape[0]
                print(f"  {done:,}/{N:,}  {el:5.0f}s  {done/max(el,1):,.0f} rows/s", flush=True)
    with open(out_path, "ab") as out, open(stmp, "rb") as sf:
        while True:
            b = sf.read(1 << 24)
            if not b: break
            out.write(b)
    os.remove(stmp)
    print(f"  wrote {out_path}  {os.path.getsize(out_path)/1024**3:.2f} GiB in {time.time()-t0:.0f}s", flush=True)

n = 0
for s in set(idx.values()):
    if s in touched: continue
    d = f"{a.out}/{s}"
    if os.path.exists(d): os.remove(d)
    os.link(f"{a.pack}/{s}", d); n += 1
print(f"hardlinked {n} untouched shards")

n_file = 0
shards = set(idx.values())
for entry in sorted(os.listdir(a.pack)):
    if entry in shards: continue
    src, dst = f"{a.pack}/{entry}", f"{a.out}/{entry}"
    if os.path.isdir(src): continue          # recipe/ etc are not needed to serve
    if os.path.exists(dst): os.remove(dst)
    try: os.link(src, dst)
    except OSError: shutil.copy2(src, dst)
    n_file += 1
print(f"carried over {n_file} non-shard entries (index/config/tokenizer)")
print("done")
