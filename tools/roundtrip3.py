#!/usr/bin/env python3
"""Round-trip check: MXINT-3 tables vs fp8 source. rel_L2 on 4096 sample rows/layer."""
import struct, json, torch

pack = "/mnt/nvme0/bigmodels/dsv41-engram-q3"
src = "/mnt/nvme0/bigmodels/dsv41-exl3-2.0bpw"

def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))

for shard, L in [("model-00047-of-00048.safetensors", 1), ("model-00048-of-00048.safetensors", 14)]:
    wn, sn = f"layers.{L}.engram.embed.weight", f"layers.{L}.engram.embed.scale"
    off, hdr = header(f"{pack}/{shard}")
    soff, shdr = header(f"{src}/{shard}")
    w = hdr[wn]
    ws = shdr[wn]
    N, cols = ws["shape"]
    R = 4096
    with open(f"{pack}/{shard}", "rb") as f:
        f.seek(off + w["data_offsets"][0])
        wb = f.read(R * (cols * 3 // 8))
        f.seek(off + hdr[sn]["data_offsets"][0])
        sb = f.read(R * (cols // 32))
    with open(f"{src}/{shard}", "rb") as f:
        f.seek(soff + ws["data_offsets"][0])
        wf = torch.frombuffer(bytearray(f.read(R * cols)), dtype=torch.uint8)
        f.seek(soff + shdr[sn]["data_offsets"][0])
        sf = torch.frombuffer(bytearray(f.read(R * (cols // 32))), dtype=torch.uint8)
    w32 = torch.frombuffer(bytearray(wb), dtype=torch.uint8).view(R, cols // 8, 3).to(torch.int32)
    v24 = w32[:, :, 0] | (w32[:, :, 1] << 8) | (w32[:, :, 2] << 16)
    bits = torch.stack([(v24 >> (3 * k)) & 7 for k in range(8)], dim=2).to(torch.int16) - 4
    scale = (torch.frombuffer(bytearray(sb), dtype=torch.uint8)
             .view(torch.float8_e4m3fn).to(torch.float32).view(R, cols // 32) / 8.0)
    approx = (bits.view(R, cols // 32, 32).to(torch.float32) * scale[:, :, None]).reshape(R, cols)
    ref = (wf.view(torch.float8_e4m3fn).to(torch.float32).view(R, cols // 32, 32)
           * sf.view(torch.float8_e8m0fnu).to(torch.float32).view(R, cols // 32)[:, :, None]).reshape(R, cols)
    rel = (approx - ref).norm() / ref.norm()
    print(f"{shard[:20]} rows={N:,} rel_L2={rel.item():.4f}", flush=True)
