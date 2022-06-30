#!/usr/bin/env sh
set -e
wget -O Checkpoints.zip.001 https://mediastore.rz.uni-augsburg.de/get/Cf9ARJEm2c/
wget -O Checkpoints.zip.002 https://mediastore.rz.uni-augsburg.de/get/DhLDsWJ45O/
wget -O Checkpoints.zip.003 https://mediastore.rz.uni-augsburg.de/get/52zXtxxetM/
wget -O Checkpoints.zip.004 https://mediastore.rz.uni-augsburg.de/get/oRVtbrVgAL/
wget -O Checkpoints.zip.005 https://mediastore.rz.uni-augsburg.de/get/nMytniYeo9/
wget -O Checkpoints.zip.006 https://mediastore.rz.uni-augsburg.de/get/zfKdnbnB5p/

cat Checkpoints.zip.* > Checkpoints.zip
rm Checkpoints.zip.*
