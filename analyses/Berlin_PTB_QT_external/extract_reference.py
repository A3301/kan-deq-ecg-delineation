"""Extract Table 4 from the DOCX conversion of Christov et al. supplementary DOC."""
import csv
import io
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

SOURCE=Path(__file__).resolve().parent/"source"
NS={"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

def reference_bytes():
    with ZipFile(SOURCE/"reference.docx") as doc:
        tree=ET.fromstring(doc.read("word/document.xml"))
    tables=tree.findall(".//w:body/w:tbl",NS)
    assert len(tables)==1
    rows=[]
    for tr in tables[0].findall("w:tr",NS):
        rows.append(["".join(n.text or "" for n in cell.findall(".//w:t",NS)).strip()
                     for cell in tr.findall("w:tc",NS)])
    assert rows[1][2:]==["R 1","R 2","R 3","R 4","R 5","Med"]*2
    assert len(rows)==551
    data=io.StringIO(newline="")
    out=csv.writer(data)
    out.writerow(["id","q1","q2","q3","q4","q5","Q","t1","t2","t3","t4","t5","T"])
    patient=None
    for p,rid,*entries in rows[2:]:
        if p:
            patient="patient%03d"%int(re.fullmatch(r"patient\s*(\d+)",p).group(1))
        assert patient and len(entries)==12 and re.fullmatch(r"s\d+[a-z_]+",rid)
        out.writerow([f"{patient}/{rid}",*map(int,entries)])
    return data.getvalue().encode("utf8")

if __name__=="__main__":
    target=SOURCE/"manual_reference.csv"
    new=reference_bytes()
    if target.is_file():
        assert target.read_bytes()==new, "Mismatch with existing normalized reference"
    else:
        target.write_bytes(new)
    print("Validated 549 reference rows, 12 numerical columns per row.")
