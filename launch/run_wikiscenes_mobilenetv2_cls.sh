#!/bin/bash

## Your values here:
#
EXP=
RUN_ID=
#
##

#
# Script
#

LOG_DIR=logs/${DS}/${EXP}
CMD="python train.py --cfg ./configs/wiki_mobilenet_cls.yaml --exp $EXP --dataset wikiscenes_corr --run $RUN_ID --loss_3d  0.3 --normalize_feature --feature decoder --use_contrastive --num_negative 16 --contrastive_tau 0.07 --workers 0 --lr_milestones 6 --use_contrastive_easy"
LOG_FILE=$LOG_DIR/${RUN_ID}.log

if [ ! -d "$LOG_DIR" ]; then
  echo "Creating directory $LOG_DIR"
  mkdir -p $LOG_DIR
fi

echo $CMD
echo "LOG: $LOG_FILE"

nohup $CMD > $LOG_FILE 2>&1 &
sleep 1
tail -f $LOG_FILE
