import re
from project_paths import ROOT_DIR

with (ROOT_DIR / "docs" / "reference" / "page_2014.html").open(encoding='utf-8') as f:
    html = f.read()

links = re.findall(r'href=[\'"]([^\'"]+)[\'"]', html, re.I)
for l in links:
    if any(k in l.lower() for k in ['xpt', 'zip', 'llcp', 'data']):
        print(l)
