#!/bin/bash

for filename in ./bundled_pip_dependencies/*.tar.gz; do
    [ -e "$filename" ] || continue
    pip install "$filename" --ignore-installed
done
