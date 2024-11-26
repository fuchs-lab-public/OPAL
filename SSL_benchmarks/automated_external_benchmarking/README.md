# Automated External Benchmarking

## Docker Container

## `inference.py` Script

This script should run within the container provided by the user.
The script should accept two arguments:
- `input` path to input csv file which lists the slides to encode. The input csv files should contain two columns: `slide` a unique slide id which will also be used to name the tensors; `slide_path` the path to the slide file.
- `output`: path to output directory.
For each slide in the input csv, a `.pth` binary tensor file should be generated in the `output` directory.
We provide an [example script]() which the users can modify to their needs.

