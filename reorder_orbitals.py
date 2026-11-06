#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reorder_orbitals.py
===================

Reorder the 76 active-space orbitals of the FeMoco FCIDUMP so that orbitals
that belong to the same atom are contiguous.

The active space is CAS(113e,76o).  The original FCIDUMP follows the orbital
order produced by the genetic-algorithm (GA) localization of BLOCK; the
corresponding orbital pictures are in ``MOs_plots/``.  After a visual
re-inspection of those pictures the orbitals were reclassified and the new
order below was proposed.  Every entry of ``GROUPS`` lists, in the *new*
order, the original (0-based) orbital indices.  Orbitals listed in the same
group belong to the same atom (or to the same "terminal S 3p" pair).

The FCIDUMP itself uses 1-based orbital labels, so the mapping applied to the
integrals is

    new_orbital k (0-based)  <->  old_orbital PERM[k] (0-based)

i.e.  h'[I,J] = h[p_I,p_J]  and  (I J|K L)' = (p_I p_J|p_K p_L)
with p_X = PERM[X] + 1 in 1-based FCIDUMP numbering.

The output file keeps the canonical FCIDUMP layout
(header, two-electron block, one-electron block, nuclear energy), so it can be
read by PySCF, BLOCK, Molpro, etc.  The numerical values are copied verbatim
from the input file.

Usage
-----
    python3 reorder_orbitals.py                       # FCIDUMP.zip -> FCIDUMP_reordered(.zip)
    python3 reorder_orbitals.py --source FCIDUMP --output FCIDUMP_reordered
    python3 reorder_orbitals.py --verify              # also re-read and cross-check
    python3 reorder_orbitals.py --print-map           # print the new -> old mapping
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import zipfile

# --------------------------------------------------------------------------
# 1. The new orbital order
# --------------------------------------------------------------------------
# Each entry: (group label, chemical character, [old 0-based orbital indices]).
# The groups are concatenated in the order given here to build the new order.
#
# NOTE: the group labels S1..S11 / Fe1..Fe7 / C1 / Mo1 are *group* labels that
# follow the new (atom-contiguous) orbital order.  They are the labels used in
# README.md; they do not necessarily coincide with the atom indices of the
# geometry block.  Fe1 is the group identified as "Fe1 3d" in the orbital
# plots.
GROUPS = [
    ("S1",  "terminal S 3p",     [0, 1]),
    ("Fe1", "Fe 3d",             [2, 3, 4, 5, 6]),
    ("S2",  "S 3p",              [7, 12, 14]),
    ("S3",  "S 3p",              [8, 13, 15]),
    ("S4",  "S 3p",              [9, 10, 11]),
    ("Fe2", "Fe 3d",             [16, 17, 18, 19, 20]),
    ("Fe3", "Fe 3d",             [21, 22, 23, 24, 25]),
    ("Fe4", "Fe 3d",             [26, 27, 28, 29, 30]),
    ("S5",  "S 3p",              [31, 32, 34]),
    ("S6",  "S 3p",              [33, 39, 40]),
    ("S7",  "S 3p",              [41, 42, 43]),
    ("C1",  "C 2s2p",            [35, 36, 37, 38]),
    ("Fe5", "Fe 3d",             [44, 45, 46, 47, 48]),
    ("Fe6", "Fe 3d",             [49, 50, 51, 52, 53]),
    ("Fe7", "Fe 3d",             [54, 55, 56, 57, 58]),
    ("S8",  "S 3p",              [59, 60, 63]),
    ("S9",  "S 3p",              [61, 62, 67]),
    ("S10", "S 3p",              [64, 66, 68]),
    ("Mo1", "Mo 4d",             [65, 69, 70, 71, 72]),
    ("S11", "S 3p",              [73, 74, 75]),
]

# Flatten: PERM[new] = old
PERM = [old for _label, _char, idxs in GROUPS for old in idxs]

# old -> new
INV = [0] * len(PERM)
for _new, _old in enumerate(PERM):
    INV[_old] = _new


# --------------------------------------------------------------------------
# 2. Small helpers
# --------------------------------------------------------------------------

HEADER_KEYS = ("NORB", "NELEC", "MS2", "ORBSYM", "ISYM")


def read_source(path: str) -> str:
    """Return the text of an FCIDUMP, either a plain file or a .zip archive."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if len(names) != 1:
                raise RuntimeError(
                    f"{path}: expected exactly one file in the archive, got {names}"
                )
            raw = zf.read(names[0])
            print(f"[read] {path} :: {names[0]}  ({len(raw)/1e6:.1f} MB)")
            return raw.decode("ascii", errors="strict")
    with open(path, "r", encoding="ascii") as fh:
        text = fh.read()
    print(f"[read] {path}  ({len(text)/1e6:.1f} MB)")
    return text


def split_header(text: str):
    """Split the leading &FCI ... &END header from the body; return (hdr, body)."""
    m = re.search(r"&END[^\n]*\n?", text, re.IGNORECASE)
    if not m:
        raise RuntimeError("FCIDUMP header does not contain &END")
    end = m.end()
    return text[:end], text[end:]


def parse_header(header: str) -> dict:
    info = {}
    for key in HEADER_KEYS:
        m = re.search(rf"{key}\s*=\s*([^\n,]+)", header, re.IGNORECASE)
        if m:
            info[key] = m.group(1).strip()
    info["NORB"] = int(info["NORB"])
    info["NELEC"] = int(info["NELEC"])
    info["MS2"] = int(info.get("MS2", 0))
    info["ISYM"] = int(info.get("ISYM", 1))
    info["ORBSYM"] = [int(x) for x in re.findall(r"-?\d+", info.get("ORBSYM", ""))]
    if len(info["ORBSYM"]) != info["NORB"]:
        info["ORBSYM"] = [1] * info["NORB"]
    return info


def fmt_header(info: dict) -> str:
    out = [
        " &FCI NORB=%4d,NELEC=%3d,MS2=%d,\n"
        % (info["NORB"], info["NELEC"], info["MS2"]),
        "  ORBSYM=%s,\n" % ",".join(str(x) for x in info["ORBSYM"]),
        "  ISYM=%d,\n" % info["ISYM"],
        " &END\n",
    ]
    return "".join(out)


def pair_index(p: int, q: int) -> int:
    """Canonical 0-based block index of a pair (order-independent)."""
    if p < q:
        p, q = q, p
    return p * (p + 1) // 2 + q


def pair_table(norb: int):
    """Return lists (I_of_block, J_of_block) for block index 0..npair-1."""
    ii, jj = [], []
    for i in range(norb):
        for j in range(i + 1):
            ii.append(i)
            jj.append(j)
    return ii, jj


# --------------------------------------------------------------------------
# 3. Reordering
# --------------------------------------------------------------------------

def reorder(text: str, perm):
    """Return (new_text, stats).  ``perm[new] = old``."""
    header, body = split_header(text)
    info = parse_header(header)
    norb = info["NORB"]

    if len(perm) != norb:
        raise ValueError(f"permutation has {len(perm)} entries, but NORB={norb}")
    if sorted(perm) != list(range(norb)):
        raise ValueError("PERM is not a permutation of 0..NORB-1")

    inv = [0] * norb
    for new, old in enumerate(perm):
        inv[old] = new

    # the point group of every orbital is carried along with the orbital
    info["ORBSYM"] = [info["ORBSYM"][old] for old in perm]

    npair = norb * (norb + 1) // 2
    nint2 = npair * (npair + 1) // 2

    two = [None] * nint2   # value strings, indexed by canonical position
    one = [None] * npair   # one-electron value strings
    ecore = None

    n_read = n_two = n_one = 0

    for line in body.splitlines():
        if not line.strip():
            continue
        tok = line.split()
        if len(tok) != 5:
            raise RuntimeError(f"malformed FCIDUMP line: {line!r}")
        v = tok[0]
        i, j, k, l = (int(t) for t in tok[1:5])
        n_read += 1

        if k != 0:  # two-electron integral
            b1 = pair_index(inv[i - 1], inv[j - 1])
            b2 = pair_index(inv[k - 1], inv[l - 1])
            if b1 < b2:
                b1, b2 = b2, b1
            pos = b1 * (b1 + 1) // 2 + b2
            if two[pos] is not None:
                raise RuntimeError(
                    f"duplicate two-electron integral at canonical position {pos}"
                )
            two[pos] = v
            n_two += 1
        elif j != 0:  # one-electron integral (k == l == 0)
            b = pair_index(inv[i - 1], inv[j - 1])
            if one[b] is not None:
                raise RuntimeError(f"duplicate one-electron integral at block {b}")
            one[b] = v
            n_one += 1
        else:  # nuclear energy
            ecore = v

    if ecore is None:
        raise RuntimeError("no nuclear-energy line (0 0 0 0) found")
    missing2 = sum(1 for x in two if x is None)
    missing1 = sum(1 for x in one if x is None)
    if missing2 or missing1:
        raise RuntimeError(
            f"incomplete FCIDUMP: {missing2} two-electron and {missing1} "
            f"one-electron integrals missing"
        )

    # ---- write the new file, in canonical FCIDUMP order ------------------
    ii, jj = pair_table(norb)
    pstr = ["%d %d" % (ii[b] + 1, jj[b] + 1) for b in range(npair)]

    chunks = [fmt_header(info)]
    buf = []
    pos = 0
    for b1 in range(npair):
        head = pstr[b1]
        for b2 in range(b1 + 1):
            buf.append(two[pos] + " " + head + " " + pstr[b2] + "\n")
            pos += 1
            if len(buf) >= 200_000:
                chunks.append("".join(buf))
                buf.clear()
    chunks.append("".join(buf))
    buf.clear()

    for b in range(npair):
        buf.append(one[b] + "  " + pstr[b] + "  0  0\n")
    chunks.append("".join(buf))

    chunks.append(ecore + " 0 0 0 0\n")
    new_text = "".join(chunks)

    stats = dict(
        norb=norb,
        npair=npair,
        nint2=nint2,
        n_read=n_read,
        n_two=n_two,
        n_one=n_one,
        ecore=ecore,
    )
    return new_text, stats


# --------------------------------------------------------------------------
# 4. Optional verification
# --------------------------------------------------------------------------

def _collect(path: str, mapping=None):
    """Read an FCIDUMP and return canonical integral arrays in a common
    (target) orbital numbering.

    ``mapping`` (optional) maps an old 0-based orbital index to the target
    0-based orbital index; ``None`` means identity.
    """
    import numpy as np

    text = read_source(path)
    header, body = split_header(text)
    info = parse_header(header)
    norb = info["NORB"]
    if mapping is None:
        mapping = list(range(norb))
    if len(mapping) != norb:
        raise RuntimeError(f"mapping length {len(mapping)} != NORB {norb}")

    k2, v2, k1, v1 = [], [], [], []
    ecore = None
    for line in body.splitlines():
        tok = line.split()
        if len(tok) != 5:
            continue
        i, j, k, l = (int(t) - 1 for t in tok[1:5])  # 0-based
        if k >= 0:  # two-electron
            b1 = pair_index(mapping[i], mapping[j])
            b2 = pair_index(mapping[k], mapping[l])
            if b1 < b2:
                b1, b2 = b2, b1
            k2.append(b1 * (b1 + 1) // 2 + b2)
            v2.append(float(tok[0]))
        elif j >= 0:  # one-electron
            k1.append(pair_index(mapping[i], mapping[j]))
            v1.append(float(tok[0]))
        else:
            ecore = tok[0]

    return (
        np.asarray(k2, dtype=np.int64),
        np.asarray(v2, dtype=np.float64),
        np.asarray(k1, dtype=np.int64),
        np.asarray(v1, dtype=np.float64),
        ecore,
        norb,
    )


def verify(old_path: str, new_path: str, perm=None):
    """Independent cross-check of a reordered FCIDUMP.

    Reads the original file with old->new index mapping and the reordered file
    with the identity mapping, then requires the two canonical
    (key, value) sets to be bit-identical.
    """
    import numpy as np

    if perm is None:
        perm = PERM
    inv = [0] * len(perm)
    for new, old in enumerate(perm):
        inv[old] = new

    print("[verify] reading original ...")
    k2o, v2o, k1o, v1o, eo, norb = _collect(old_path, mapping=inv)
    print("[verify] reading reordered ...")
    k2n, v2n, k1n, v1n, en, _ = _collect(new_path)

    npair = norb * (norb + 1) // 2
    nint2 = npair * (npair + 1) // 2
    if eo != en:
        raise SystemExit(f"[verify] FAILED: ECORE {eo} != {en}")
    if len(k2o) != nint2:
        raise SystemExit(f"[verify] FAILED: {len(k2o)} two-electron integrals, expected {nint2}")
    if len(k1o) != npair:
        raise SystemExit(f"[verify] FAILED: {len(k1o)} one-electron integrals, expected {npair}")

    for name, ko, vo, kn, vn in (
        ("two-electron", k2o, v2o, k2n, v2n),
        ("one-electron", k1o, v1o, k1n, v1n),
    ):
        if not np.array_equal(np.sort(ko), np.sort(kn)):
            raise SystemExit(f"[verify] FAILED: {name} index set differs")
        o = np.argsort(ko, kind="stable")
        p = np.argsort(kn, kind="stable")
        if not np.array_equal(ko[o], kn[p]):
            raise SystemExit(f"[verify] FAILED: {name} key mismatch")
        if not np.array_equal(vo[o], vn[p]):
            bad = np.flatnonzero(vo[o] != vn[p])[:5]
            raise SystemExit(f"[verify] FAILED: {name} value mismatch at {bad}")

    print(
        "[verify] OK: %d two-electron + %d one-electron integrals, ECORE=%s"
        % (len(k2o), len(k1o), eo)
    )


# --------------------------------------------------------------------------
# 5. Command line
# --------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Reorder the FeMoco active-space orbitals to be atom-contiguous."
    )
    p.add_argument(
        "--source",
        default=None,
        help="input FCIDUMP or FCIDUMP.zip (default: FCIDUMP.zip, else FCIDUMP)",
    )
    p.add_argument("--output", default="FCIDUMP_reordered", help="output file")
    p.add_argument("--no-zip", action="store_true", help="do not create a .zip archive")
    p.add_argument("--verify", action="store_true", help="cross-check the output")
    p.add_argument("--print-map", action="store_true", help="print the new->old map")
    args = p.parse_args(argv)

    if args.print_map:
        new = 0
        for label, char, idxs in GROUPS:
            for old in idxs:
                print(f"{new:3d}  {old:3d}  {label:<5s} {char}")
                new += 1
        return 0

    src = args.source
    if src is None:
        src = "FCIDUMP.zip" if os.path.exists("FCIDUMP.zip") else "FCIDUMP"
    if not os.path.exists(src):
        p.error(f"input file not found: {src}")

    t0 = time.time()
    text = read_source(src)
    new_text, stats = reorder(text, PERM)
    with open(args.output, "w", encoding="ascii", newline="\n") as fh:
        fh.write(new_text)
    print(
        "[write] %s  (%.1f MB, %d two-electron + %d one-electron integrals, "
        "ECORE=%s) in %.1f s"
        % (
            args.output,
            len(new_text) / 1e6,
            stats["n_two"],
            stats["n_one"],
            stats["ecore"],
            time.time() - t0,
        )
    )

    if args.verify:
        verify(src, args.output)

    if not args.no_zip:
        zpath = args.output + ".zip"
        with zipfile.ZipFile(
            zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as zf:
            zf.write(args.output, arcname=os.path.basename(args.output))
        print(f"[zip]   {zpath}  ({os.path.getsize(zpath)/1e6:.1f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
