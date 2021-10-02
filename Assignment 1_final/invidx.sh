#!/bin/bash
mkdir tempfiles
python3 invidx_cons.py $1 $2 $3 $4 $5
rm -r tempfiles