#!/bin/bash

nohup python main.py > output.log 2>&1 &
echo $! > pid.tmp
tail -f output.log