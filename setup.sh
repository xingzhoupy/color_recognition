#!/usr/bin/env bash
docker run -d -v $PWD/color_recognition:/mnt/color_recognition -p 9000:5000 color_recognition sh /mnt/color_recognition/app.sh