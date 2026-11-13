"""Build a deterministic distribution ZIP; local verification artifacts stay out."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();package=args.package.resolve()
    target=package/'SINCON_monday_delivery.zip'
    if target.exists():raise FileExistsError('Archive exists; use a new package copy')
    excluded={'FILE_LIST.tsv','DISTRIBUTION.json','SINCON_monday_delivery.zip'}
    files=[]
    for p in sorted(package.rglob('*')):
        rel=p.relative_to(package)
        if not p.is_file() or p.name in excluded or '__pycache__' in rel.parts or any(x.startswith('.') for x in rel.parts):continue
        if rel.parts[0] in ['outputs','verification','.validation']:continue
        if rel.parts[0] in ['data','input'] and p.suffix.lower()!='.md':continue
        files.append(p)
    stream=io.StringIO(newline='');writer=csv.writer(stream,delimiter='\t',lineterminator='\n')
    writer.writerow(['path','bytes','sha256'])
    for p in files:writer.writerow([str(p.relative_to(package)),p.stat().st_size,sha(p)])
    listing=package/'FILE_LIST.tsv';listing.write_text(stream.getvalue())
    with ZipFile(target,'x',ZIP_DEFLATED,compresslevel=9) as archive:
        for p in sorted([*files,listing]):
            rel=Path('monday_delivery')/p.relative_to(package)
            info=ZipInfo(str(rel),date_time=(2026,11,13,0,0,0));info.compress_type=ZIP_DEFLATED
            info.external_attr=(0o100644<<16)
            archive.writestr(info,p.read_bytes(),compress_type=ZIP_DEFLATED,compresslevel=9)
    distribution=dict(archive=target.name,bytes=target.stat().st_size,sha256=sha(target),
        archive_members=len(files)+1,files_listed=len(files),file_list_sha256=sha(listing),
        root_directory='monday_delivery',official_csv_included=False,virtualenv_and_wheels_included=False,
        self_referential_exclusions=sorted(excluded),final_archive_execution=None)
    (package/'DISTRIBUTION.json').write_text(json.dumps(distribution,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(distribution,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
