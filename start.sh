#!/bin/bash

nohup python jukebox-webui.py > output.log 2>&1 &
echo $! > pid.tmp
tail -f output.log