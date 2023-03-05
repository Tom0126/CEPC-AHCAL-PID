#!/bin/bash
for((i=14578336;i<14578353;i++))
do
condor_rm $i
done
