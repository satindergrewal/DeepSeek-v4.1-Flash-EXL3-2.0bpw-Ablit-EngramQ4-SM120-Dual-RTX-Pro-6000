#!/usr/bin/env python3
"""Verify ablit graft shard placement + build hybrid ablit+q4engram pack."""
import json, os

src = "/mnt/nvme0/bigmodels/dsv41-exl3-2.0bpw"
ablit = "/mnt/nvme0/bigmodels/dsv41-exl3-2.0bpw-ablit"
q4 = "/mnt/nvme0/bigmodels/dsv41-engram-q4"
out = "/mnt/nvme0/bigmodels/dsv41-ablit-engram-q4"
S47 = {"model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors"}

idx = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
wo_b = {s for n, s in idx.items() if ".wo_b." in n}
engram = {s for n, s in idx.items() if "engram.embed." in n}
print("wo_b shards:", len(wo_b), sorted(wo_b)[:4], "...")
print("overlap wo_b x 47/48:", sorted(wo_b & S47))
print("engram shards:", sorted(engram))
assert engram == S47, "engram tensor shard layout changed?"

# spot-verify the graft actually differs from stock: hash one wo_b tensor region
import hashlib, struct
def tensor_digest(pack, name):
    with open(f"{pack}/model.safetensors.index.json") as f:
        im = json.load(f)["weight_map"]
    shard = im[name]
    with open(f"{pack}/{shard}", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    meta = hdr[name]
    off = 8 + n + meta["data_offsets"][0]
    size = meta["data_offsets"][1] - meta["data_offsets"][0]
    with open(f"{pack}/{shard}", "rb") as f:
        f.seek(off)
        h = hashlib.sha256()
        remaining = min(size, 1 << 20)
        while remaining:
            b = f.read(min(1 << 20, remaining))
            h.update(b)
            remaining -= len(b)
    return h.hexdigest()[:16]

import re
cand = sorted(n for n in idx if ".wo_b." in n and re.search(r"layers\.(1[0-9]|2[0-9]|3[0-5])\.", n))
pick = cand[0]
d1 = tensor_digest(ablit, pick)
d2 = tensor_digest(src, pick)
print(f"graft check {pick[:60]}: ablit={d1} src={d2} differ={d1 != d2}")
assert d1 != d2, "ablit tensor identical to stock - graft missing?"

# build hybrid: all shards from ablit except 47/48 from q4; non-shard load files from ablit
os.makedirs(out, exist_ok=True)
for i in range(1, 49):
    name = f"model-{i:05d}-of-00048.safetensors"
    dst = f"{out}/{name}"
    if os.path.exists(dst):
        os.remove(dst)
    srcfile = f"{q4}/{name}" if name in S47 else f"{ablit}/{name}"
    os.link(srcfile, dst)
n = 0
shards = {f"model-{i:05d}-of-00048.safetensors" for i in range(1, 49)}
for entry in os.listdir(ablit):
    if entry in shards or os.path.isdir(f"{ablit}/{entry}"):
        continue
    if not (entry.endswith((".json", ".model", ".txt")) or "tokenizer" in entry):
        continue
    dst = f"{out}/{entry}"
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(f"{ablit}/{entry}", dst)
    except OSError:
        import shutil
        shutil.copy2(f"{ablit}/{entry}", dst)
    n += 1
print("hybrid pack built at", out, "| carried", n, "config files")
for name in sorted(os.listdir(out))[:4]:
    print("  ", name)
