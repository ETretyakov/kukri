"""Generate ranges from unicode_categories 0.1.1 (the HF Rust dependency)."""

import argparse
import hashlib
import re
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("tables")
p.add_argument("output")
a = p.parse_args()
raw = Path(a.tables).read_text()
tables = {
    n: [int(x, 16) for x in re.findall(r"\\u\{([0-9A-Fa-f]+)\}", s)]
    for n, s in re.findall(r"pub static (\w+).*?= &\[(.*?)\];", raw, re.S)
}
text = (
    "// Code generated from unicode_categories 0.1.1; DO NOT EDIT.\n// See UNICODE_LICENSE. Input SHA256: "
    + hashlib.sha256(raw.encode()).hexdigest()
    + "\npackage tokenbudget\n\n"
)
for dest, names in [
    ("bertOther", ["OTHER_CONTROL", "OTHER_FORMAT"]),
    ("bertMark", ["MARK_NONSPACING"]),
    ("bertPunctuation", [n for n in tables if n.startswith("PUNCTUATION_")]),
]:
    vals = sorted({x for n in names for x in tables[n]})
    ranges = []
    for x in vals:
        if ranges and ranges[-1][1] + 1 == x:
            ranges[-1][1] = x
        else:
            ranges.append([x, x])
    text += (
        "var "
        + dest
        + " = [][2]rune{\n"
        + "".join("{0x%x,0x%x},\n" % tuple(v) for v in ranges)
        + "}\n\n"
    )
text += "func inRanges(r rune, ranges [][2]rune) bool {\n lo,hi:=0,len(ranges)\n for lo<hi {mid:=(lo+hi)/2;if r<ranges[mid][0]{hi=mid}else if r>ranges[mid][1]{lo=mid+1}else{return true}}\n return false\n}\n"
Path(a.output).write_text(text)
