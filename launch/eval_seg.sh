#!/bin/bash

DATASET=wikiscenes_corr
FILELIST=./data/wikiscenes_val.txt

## You values here:
#
OUTPUT_DIR=./logs/masks
EXP=final
RUN_ID=ours
#
##


LISTNAME=`basename $FILELIST .txt`

SAVE_DIR=$OUTPUT_DIR/$DATASET/$EXP/$RUN_ID/$LISTNAME
python eval_seg.py --data ./data --filelist $FILELIST --masks $SAVE_DIR
