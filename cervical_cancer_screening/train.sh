{\rtf1\ansi\ansicpg1252\cocoartf1504\cocoasubrtf810
{\fonttbl\f0\fnil\fcharset0 HelveticaNeue;}
{\colortbl;\red255\green255\blue255;\red53\green53\blue53;}
{\*\expandedcolortbl;;\cssrgb\c27059\c27059\c27059;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\deftab560
\pard\pardeftab560\slleading20\partightenfactor0

\f0\fs24 \cf2 PRETRAINED_CHECKPOINT_DIR=~/pretrain\
TRAIN_DIR=~/runs/inception_v3_2\
DATASET_DIR=~/tfrecord\
python ~/models/slim/train_image_classifier.py \\\
	  --train_dir=$\{TRAIN_DIR\} \\\
	  --dataset_name=cervix \\\
	  --dataset_split_name=train \\\
	  --dataset_dir=$\{DATASET_DIR\} \\\
	  --model_name=inception_v3 \\\
	  --checkpoint_path=$\{PRETRAINED_CHECKPOINT_DIR\}/inception_v3.ckpt \\\
	  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \\\
	  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \\\
	  --max_number_of_steps=1000 \\\
	  --batch_size=32 \\\
	  --learning_rate=0.01 \\\
	  --learning_rate_decay_type=fixed \\\
	  --save_interval_secs=60 \\\
	  --save_summaries_secs=60 \\\
	  --log_every_n_steps=100 \\\
	  --optimizer=rmsprop \\\
	  --weight_decay=0.00004\
\
python ~/models/slim/train_image_classifier.py \\\
	  --train_dir=$\{TRAIN_DIR\}/all \\\
	  --dataset_name=cervix \\\
	  --dataset_split_name=train \\\
	  --dataset_dir=$\{DATASET_DIR\} \\\
	  --model_name=inception_v3 \\\
	  --checkpoint_path=$\{TRAIN_DIR\} \\\
	  --max_number_of_steps=500 \\\
	  --batch_size=32 \\\
	  --learning_rate=0.0001 \\\
	  --learning_rate_decay_type=fixed \\\
	  --save_interval_secs=60 \\\
	  --save_summaries_secs=60 \\\
	  --log_every_n_steps=10 \\\
	  --optimizer=rmsprop \\\
	  --weight_decay=0.00004\
\
\pard\pardeftab560\slleading20\partightenfactor0
\cf2 \ul \ulc2 get test predictions\
\ulnone TRAIN_DIR=~/runs/inception_v3_2/all\
DATASET_DIR=~/tfrecord\
python ~/models/slim/get_predictions.py \\\
	  --checkpoint_path=$\{TRAIN_DIR\} \\\
	  --eval_dir=$\{TRAIN_DIR\} \\\
	  --dataset_name=cervix \\\
	  --dataset_split_name=test \\\
	  --dataset_dir=$\{DATASET_DIR\} \\\
	  --model_name=inception_v3}