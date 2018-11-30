import json
import glob
import os

data = []

for fname in glob.glob(os.path.join('*.json')):
    with open(fname) as f:
        jf = json.loads(f.read())
        data.extend(jf)

with open('corpus.json', 'w') as f:
    print(json.dumps(data, indent=4), file=f)