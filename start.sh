#!/bin/bash

nohup python -u main.py > output.log 2>&1 &
echo $! > pid.tmp
tail -f output.log