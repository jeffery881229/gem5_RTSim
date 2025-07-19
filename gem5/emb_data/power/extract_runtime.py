#!/usr/bin/env python3
import glob, re

pattern = re.compile(r'Processor:[\s\S]*?Runtime Dynamic\s*=\s*([\d.]+)')

for fname in sorted(glob.glob("*.log")):
    with open(fname, 'r') as f:
        txt = f.read()
    m = pattern.search(txt)
    if m:
        print(f"{fname} → {m.group(1)}")
    else:
        print(f"{fname} → NOT FOUND")
