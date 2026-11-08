import xml.etree.ElementTree as ET
from project_paths import ROOT_DIR
from pathlib import Path

reference_dir = ROOT_DIR / "docs" / "reference"
tree = ET.parse(reference_dir / "xie2019_pmc6795062.xml")
root = tree.getroot()

def get_text(elem):
    return ' '.join(''.join(elem.itertext()).split())

output_path = reference_dir / 'paper_extracted_text.txt'
with output_path.open('w', encoding='utf-8') as out:
    for sec in root.iter('sec'):
        title = sec.find('title')
        title_text = get_text(title) if title is not None else 'Untitled'
        out.write(f"\n{'='*20} {title_text} {'='*20}\n")
        for p in sec.iter('p'):
            out.write(get_text(p) + "\n\n")

    # Extract tables
    for tbl in root.iter('table-wrap'):
        tbl_id = tbl.get('id', '')
        tbl_label = tbl.find('label')
        tbl_caption = tbl.find('caption')
        label_text = get_text(tbl_label) if tbl_label is not None else ''
        caption_text = get_text(tbl_caption) if tbl_caption is not None else ''
        out.write(f"\n{'*'*20} TABLE: {tbl_id} {label_text}: {caption_text} {'*'*20}\n")
        # print rows
        for tr in tbl.iter('tr'):
            row = [get_text(c) for c in list(tr)]
            out.write(' | '.join(row) + "\n")

    # Extract app-group (appendix)
    for app in root.iter('app'):
        out.write(f"\n{'#'*20} APPENDIX {'#'*20}\n")
        out.write(get_text(app) + "\n")

print(f"Extracted paper text to {output_path}")
