#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import math
import time
import keras
import random
import socket
import shutil
import subprocess
import numpy as np
import tensorflow as tf
import keras.backend as K
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import Callback

HOST = '127.0.0.1'
PORT = 12012

MAX_FILE_SIZE = 10000
MAX_BITMAP_SIZE = 2000
round_cnt = 0
seed = 12
np.random.seed(seed)
random.seed(seed)
tf.set_random_seed(seed)
seed_list = glob.glob('./seeds/*')
new_seeds = glob.glob('./seeds/id_*')
SPLIT_RATIO = len(seed_list)
argvv = sys.argv[1:]

def process_data():
    global MAX_BITMAP_SIZE, MAX_FILE_SIZE, SPLIT_RATIO, seed_list, new_seeds

    seed_list = glob.glob('./seeds/*')
    seed_list.sort()
    SPLIT_RATIO = len(seed_list)
    np.random.shuffle(seed_list)
    new_seeds = glob.glob('./seeds/id_*')

    call = subprocess.check_output
    cwd = os.getcwd()
    max_file_name = call(['ls', '-S', cwd + '/seeds/']).decode('utf8').split('\n')[0].rstrip('\n')
    MAX_FILE_SIZE = os.path.getsize(cwd + '/seeds/' + max_file_name)

    os.makedirs("./bitmaps", exist_ok=True)
    os.makedirs("./splice_seeds", exist_ok=True)
    os.makedirs("./vari_seeds", exist_ok=True)
    os.makedirs("./crashes", exist_ok=True)

    raw_bitmap = {}
    tmp_cnt = []
    out = ''
    for f in seed_list:
        tmp_list = []
        try:
            if argvv[0] == './strip':
                out = call(['./afl-showmap', '-q', '-e', '-o', '/dev/stdout', '-m', '512', '-t', '500'] + argvv + [f] + ['-o', 'tmp_file'])
            else:
                out = call(['./afl-showmap', '-q', '-e', '-o', '/dev/stdout', '-m', '512', '-t', '500'] + argvv + [f])
        except subprocess.CalledProcessError:
            print("find a crash")
        for line in out.splitlines():
            edge = line.split(b':')[0]
            tmp_cnt.append(edge)
            tmp_list.append(edge)
        raw_bitmap[f] = tmp_list
    counter = Counter(tmp_cnt).most_common()

    label = [int(f[0]) for f in counter]
    bitmap = np.zeros((len(seed_list), len(label)))
    for idx, i in enumerate(seed_list):
        tmp = raw_bitmap[i]
        for j in tmp:
            if int(j) in label:
                bitmap[idx][label.index((int(j)))] = 1

    fit_bitmap = np.unique(bitmap, axis=1)
    print("data dimension" + str(fit_bitmap.shape))

    MAX_BITMAP_SIZE = fit_bitmap.shape[1]
    for idx, i in enumerate(seed_list):
        file_name = "./bitmaps/" + i.split('/')[-1]
        np.save(file_name, fit_bitmap[idx])

def generate_training_data(lb, ub):
    seed = np.zeros((ub - lb, MAX_FILE_SIZE))
    bitmap = np.zeros((ub - lb, MAX_BITMAP_SIZE))
    for i in range(lb, ub):
        tmp = open(seed_list[i], 'rb').read()
        ln = len(tmp)
        if ln < MAX_FILE_SIZE:
            tmp = tmp + (MAX_FILE_SIZE - ln) * b'\x00'
        seed[i - lb] = [j for j in bytearray(tmp)]

    for i in range(lb, ub):
        file_name = "./bitmaps/" + seed_list[i].split('/')[-1] + ".npy"
        bitmap[i - lb] = np.load(file_name)
    return seed, bitmap

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.7
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))
        print(step_decay(len(self.losses)))

def accur_1(y_true, y_pred):
    y_true = tf.round(y_true)
    pred = tf.round(y_pred)
    summ = tf.constant(MAX_BITMAP_SIZE, dtype=tf.float32)
    wrong_num = tf.subtract(summ, tf.reduce_sum(tf.cast(tf.equal(y_true, pred), tf.float32), axis=-1))
    right_1_num = tf.reduce_sum(tf.cast(tf.logical_and(tf.cast(y_true, tf.bool), tf.cast(pred, tf.bool)), tf.float32), axis=-1)
    return K.mean(tf.divide(right_1_num, tf.add(right_1_num, wrong_num)))

def train_generate(batch_size):
    global seed_list
    while 1:
        np.random.shuffle(seed_list)
        for i in range(0, SPLIT_RATIO, batch_size):
            if (i + batch_size) > SPLIT_RATIO:
                x, y = generate_training_data(i, SPLIT_RATIO)
                x = x.astype('float32') / 255
            else:
                x, y = generate_training_data(i, i + batch_size)
                x = x.astype('float32') / 255
            yield (x, y)

def vectorize_file(fl):
    seed = np.zeros((1, MAX_FILE_SIZE))
    tmp = open(fl, 'rb').read()
    ln = len(tmp)
    if ln < MAX_FILE_SIZE:
        tmp = tmp + (MAX_FILE_SIZE - ln) * b'\x00'
    seed[0] = [j for j in bytearray(tmp)]
    seed = seed.astype('float32') / 255
    return seed

def splice_seed(fl1, fl2, idxx):
    tmp1 = open(fl1, 'rb').read()
    ret = 1
    randd = fl2
    while ret == 1:
        tmp2 = open(randd, 'rb').read()
        if len(tmp1) >= len(tmp2):
            lenn = len(tmp2)
            head = tmp2
            tail = tmp1
        else:
            lenn = len(tmp1)
            head = tmp1
            tail = tmp2
        f_diff = 0
        l_diff = 0
        for i in range(lenn):
            if tmp1[i] != tmp2[i]:
                f_diff = i
                break
        for i in reversed(range(lenn)):
            if tmp1[i] != tmp2[i]:
                l_diff = i
                break
        if f_diff >= 0 and l_diff > 0 and (l_diff - f_diff) >= 2:
            splice_at = f_diff + random.randint(1, l_diff - f_diff - 1)
            head = list(head)
            tail = list(tail)
            tail[:splice_at] = head[:splice_at]
            with open('./splice_seeds/tmp_' + str(idxx), 'wb') as f:
                f.write(bytearray(tail))
            ret = 0
        print(f_diff, l_diff)
        randd = random.choice(seed_list)

def gen_vari_seeds(fl, idxx):
    tmp = open(fl, 'rb').read()
    for _ in range(2):
        tmp = list(tmp)
        byte_num = random.randint(1, 2)
        for _ in range(byte_num):
            byte_val = random.randint(0, 255)
            byte_idx = random.randint(0, len(tmp) - 1)
            tmp[byte_idx] = byte_val
        with open('./vari_seeds/tmp_' + str(idxx), 'wb') as f:
            f.write(bytearray(tmp))

def eval_type1_bitmap(bitmap):
    eval_bitmap = np.zeros((1, MAX_BITMAP_SIZE))
    for line in bitmap.splitlines():
        edge = line.split(b':')[0]
        if int(edge) < MAX_BITMAP_SIZE:
            eval_bitmap[0][int(edge)] = 1
    return eval_bitmap

def mutate_one_seed(model, bit):
    global round_cnt, seed_list
    for _ in range(50):
        batch_input = np.zeros((100, MAX_FILE_SIZE))
        for j in range(100):
            rand_seed = random.choice(seed_list)
            seed_vector = vectorize_file(rand_seed)
            batch_input[j] = seed_vector
        preds = model.predict(batch_input)
        top_10 = np.argsort(np.sum(preds, axis=1))[-10:]
        for k in top_10:
            seed_file = seed_list[k]
            splice_seed(seed_file, rand_seed, round_cnt)
            gen_vari_seeds(seed_file, round_cnt)
            round_cnt += 1
        bit_vector = np.zeros((1, MAX_BITMAP_SIZE))
        bit_vector[0][bit] = 1
        mutated_input = np.zeros((10, MAX_FILE_SIZE))
        for i in range(10):
            rand_seed = random.choice(seed_list)
            seed_vector = vectorize_file(rand_seed)
            seed_vector[0][bit] = 1
            mutated_input[i] = seed_vector
        mutated_preds = model.predict(mutated_input)
        top_idx = np.argmax(np.sum(mutated_preds, axis=1))
        selected_seed = mutated_input[top_idx]
        new_seed = list(bytearray(selected_seed[0]))
        new_seed[bit] = random.randint(0, 255)
        with open('./seeds/id_' + str(round_cnt), 'wb') as f:
            f.write(bytearray(new_seed))
        round_cnt += 1

def update_queue(model):
    global seed_list, new_seeds
    while True:
        if len(new_seeds) == 0:
            print("no new seed")
            time.sleep(1)
            continue
        for idx, i in enumerate(new_seeds):
            bitmap = open('./bitmaps/' + i.split('/')[-1] + '.txt').read()
            eval_bitmap = eval_type1_bitmap(bitmap)
            preds = model.predict(eval_bitmap)
            preds = np.sum(preds)
            if preds < 0.3:
                print("found new seed " + i)
                mutate_one_seed(model, idx)
            else:
                print("no new bitmap")
        new_seeds = glob.glob('./seeds/id_*')

class CopySeedDirectoryCallback(Callback):
    def __init__(self, src_directory, dst_directory, target_epoch):
        super().__init__()
        self.src_directory = src_directory
        self.dst_directory = dst_directory
        self.target_epoch = target_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.target_epoch - 1:  # epoch is 0-indexed, so target_epoch - 1
            if os.path.exists(self.dst_directory):
                shutil.rmtree(self.dst_directory)
            shutil.copytree(self.src_directory, self.dst_directory)
            print(f"Copied contents of {self.src_directory} to {self.dst_directory} at epoch {epoch + 1}")

            # Signal reaching epoch 100
            with open('epoch_100_reached', 'w') as f:
                f.write('Epoch 100 reached.')
def train(model):
    loss_history = LossHistory()
    lrate = keras.callbacks.LearningRateScheduler(step_decay)
    copy_seed_callback = CopySeedDirectoryCallback('seed', 'input_corpus', 100)
    callbacks_list = [loss_history, lrate, copy_seed_callback]

    model.fit_generator(train_generate(16),
                        steps_per_epoch=(SPLIT_RATIO / 16 + 1),
                        epochs=100,
                        verbose=1, callbacks=callbacks_list)
    # Save model and weights
    model.save_weights("hard_label.h5")

def init():
    global seed_list
    process_data()
    print('process data finished')
    input_dim = MAX_FILE_SIZE
    output_dim = MAX_BITMAP_SIZE
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(output_dim, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[accur_1])
    print("begin train")
    train(model)
    print("end train")
    update_queue(model)

if __name__ == "__main__":
    init()

