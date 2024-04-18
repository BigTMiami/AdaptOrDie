import gzip
import shutil

filename = "datasets/domain/cs/file_0.gz"
outfile = "datasets/domain/cs/file_0.xml"
with gzip.open(filename, "rb") as f_in:
    with open(outfile, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


import json

with open("datasets/domain/cs/file_0_out.xml", "r") as f:
    data = json.load(f)
