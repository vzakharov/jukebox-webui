#!/bin/bash

nohup python jukebox-webui.py > output.log 2>&1 &
tail -f output.log