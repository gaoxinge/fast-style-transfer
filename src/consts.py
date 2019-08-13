import random

GPU_DEVICE = '/gpu:0'
CPU_DEVICE = '/cpu:0'

CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2

LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
BATCH_SIZE = 4
POOLING = 'max'

CHECKPOINT_ITERATIONS = 2000

VGG_PATH = 'data/imagenet-vgg-verydeep-19.mat'
TRAIN_PATH = 'data/train2014'

TMP_DIR = '.fns_frames_%s/' % random.randint(0, 99999)
