"""Dump a .docx body to a tagged text file: one block per line-group, preserving style, runs (bold/italic/sub/sup/font), tables, images."""
import sys, zipfile, re, json
import xml.etree.ElementTree as ET

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
ns = {"w": W, "r": R, "a": A}
q = lambda t: "{%s}%s" % (W, t)

z = zipfile.ZipFile(sys.argv[1])
doc = ET.fromstring(z.read("word/document.xml"))
rels = ET.fromstring(z.read("word/_rels/document.xml.rels"))
rid = {r.get("Id"): r.get("Target") for r in rels}


def runs(p):
    out = []
    for el in p.iter():
        if el.tag == q("r"):
            rpr = el.find("w:rPr", ns)
            fl = ""
            if rpr is not None:
                if rpr.find("w:b", ns) is not None and rpr.find("w:b", ns).get(q("val"), "1") not in ("0", "false"): fl += "b"
                if rpr.find("w:i", ns) is not None and rpr.find("w:i", ns).get(q("val"), "1") not in ("0", "false"): fl += "i"
                va = rpr.find("w:vertAlign", ns)
                if va is not None: fl += {"subscript": "_", "superscript": "^"}.get(va.get(q("val")), "")
                f = rpr.find("w:rFonts", ns)
                if f is not None and "Consolas" in (f.get(q("ascii")) or ""): fl += "m"
            txt = ""
            for c in el:
                if c.tag == q("t"): txt += c.text or ""
                elif c.tag == q("tab"): txt += "\t"
                elif c.tag == q("br"): txt += "\n" if c.get(q("type")) != "page" else "<PAGEBREAK>"
            for b in el.iter("{%s}blip" % A):
                txt += "<IMG %s>" % rid.get(b.get("{%s}embed" % R))
            if txt:
                if out and out[-1][0] == fl: out[-1][1] += txt
                else: out.append([fl, txt])
        elif el.tag == q("hyperlink"):
            pass
    return out


def fmt(rs):
    s = ""
    for fl, t in rs:
        s += ("{%s|%s}" % (fl, t)) if fl else t
    return s


def para(p):
    ppr = p.find("w:pPr", ns)
    st = ""
    if ppr is not None:
        s = ppr.find("w:pStyle", ns)
        st = s.get(q("val")) if s is not None else ""
        if ppr.find("w:numPr", ns) is not None: st += "+num"
    return st, fmt(runs(p))


lines = []
body = doc.find("w:body", ns)
for el in body:
    if el.tag == q("p"):
        st, t = para(el)
        lines.append("[P:%s] %s" % (st, t))
    elif el.tag == q("tbl"):
        lines.append("[TABLE]")
        for tr in el.findall("w:tr", ns):
            cells = []
            for tc in tr.findall("w:tc", ns):
                cells.append(" / ".join(para(p)[1] for p in tc.findall(".//w:p", ns)))
            lines.append("  | " + " || ".join(cells))
        lines.append("[/TABLE]")
    elif el.tag == q("sdt"):
        lines.append("[SDT/TOC]")
open(sys.argv[2], "w", encoding="utf-8").write("\n".join(lines))
print(len(lines))
