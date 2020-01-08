#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import queue
import threading
import datetime
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from aip import AipOcr


class MP4:
    def __init__(self, video_path="", fps=0, frame_total_num=0, length=0, time_slice_len=0, frame_slice_list=None,
                 lang1="", lang2="", lang1_subtitle_loc=None, lang2_subtitle_loc=None, audio_path="", video_cap=None):
        self.video_path = video_path
        self.fps = fps
        self.frame_total_num = frame_total_num
        self.length = length
        # self.get_length()  # self.frame_total_num / self.fps
        self.time_slice_len = time_slice_len
        self.frame_slice_list = frame_slice_list
        # self.get_frame_slice_list()
        self.lang1 = lang1
        self.lang2 = lang2
        self.lang1_subtitle_loc = lang1_subtitle_loc
        self.lang2_subtitle_loc = lang2_subtitle_loc
        self.audio_path = audio_path
        self.video_cap = video_cap

    def get_length(self):
        self.length = self.frame_total_num / self.fps

    def get_frame_slice_list(self):
        self.frame_slice_list = []

        current_sec = 0
        while current_sec < self.length:
            frame_slice_start = int(current_sec * self.fps)
            current_sec += self.time_slice_len
            frame_slice_end = int(current_sec * self.fps)

            self.frame_slice_list.append([frame_slice_start, frame_slice_end])

        self.frame_slice_list[-1][1] = self.frame_total_num - 1


class MP4Slice(MP4):
    subtitle_row_num = 10  # num of stacked subtitles in images generated for ocr

    def __init__(self, mp4, frame_slice=None, lang1_subtitle_img_list=None, lang2_subtitle_img_list=None,
                 lang1_subtitle_text_list=None, lang2_subtitle_text_list=None, time_loc_list=None):
        super(MP4Slice, self).__init__(mp4.video_path, mp4.fps, mp4.frame_total_num, mp4.length, mp4.time_slice_len,
                                       mp4.frame_slice_list, mp4.lang1, mp4.lang2,
                                       mp4.lang1_subtitle_loc, mp4.lang2_subtitle_loc, mp4.audio_path, mp4.video_cap)
        # print(mp4.video_path)
        # print(mp4.fps)
        # print(mp4.frame_total_num)
        # print(mp4.length)
        # print(mp4.time_slice_len)
        # print(mp4.frame_slice_list)
        # print(mp4.lang1)
        # print(mp4.lang2)
        # print(mp4.lang1_subtitle_loc)
        # print(mp4.lang2_subtitle_loc)
        # print(mp4.audio_path)
        # print()
        self.frame_slice = frame_slice
        self.lang1_subtitle_img_list = lang1_subtitle_img_list
        self.lang2_subtitle_img_list = lang2_subtitle_img_list
        self.lang1_subtitle_text_list = lang1_subtitle_text_list
        self.lang2_subtitle_text_list = lang2_subtitle_text_list
        self.time_loc_list = time_loc_list
        self.time_slice_tuple = (self.get_time_loc(self.frame_slice[0]), self.get_time_loc(self.frame_slice[1]))

    def get_time_loc(self, current_frame):
        total_sec = int(current_frame / self.fps)

        hour = total_sec // 3600
        minute = (total_sec - hour * 3600) // 60
        second = total_sec - hour * 3600 - minute * 60

        return datetime.time(hour=hour, minute=minute, second=second).isoformat()

    def get_subtitle_img(self):
        def std_err(img_current_frame, img_last_frame=None):
            if img_last_frame is None:  # checks if the img contains a subtitle
                return (img_current_frame ** 2).sum() / img_current_frame.size * 100
            else:  # checks if the subtitle in the current frame is the same as the one in the last frame
                return ((img_current_frame - img_last_frame) ** 2).sum() / img_current_frame.size * 100

        # gets a frame slice
        self.frame_slice = self.frame_slice_list.pop(0)

        current_frame = self.frame_slice[0]
        lang1_last_img = None
        lang2_last_img = None
        lang1_subtitle_img = None
        lang2_subtitle_img = None
        self.lang1_subtitle_img_list = []
        self.lang2_subtitle_img_list = []
        self.time_loc_list = []
        current_row_num = 0
        done = False  # checks if the final frame in each slice is reached
        while current_frame <= self.frame_slice[1]:
            # jumps
            for _ in range(9):
                self.video_cap.read()
                current_frame += 1

            if current_frame >= self.frame_slice[1]:
                current_frame = self.frame_slice[1]
                done = True

            # reads a frame
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            _, frame = self.video_cap.read()
            current_frame += 1

            img = frame[:, :, 0]
            # gets lang1 subtitle
            lang1_img = img[self.lang1_subtitle_loc[0]:self.lang1_subtitle_loc[1], :]
            # gets lang2 subtitle
            lang2_img = img[self.lang2_subtitle_loc[0]:self.lang2_subtitle_loc[1], :]

            # binarization
            thresh = 220
            _, lang1_img = cv2.threshold(lang1_img, thresh, 255, cv2.THRESH_BINARY)
            _, lang2_img = cv2.threshold(lang2_img, thresh, 255, cv2.THRESH_BINARY)

            # checks if the img contains a subtitle
            if std_err(lang1_img) < 1 and std_err(lang2_img) < 1:
                continue

            # stacks the subtitles, 10 rows per image used in ocr
            if current_row_num == 0:
                lang1_subtitle_img = lang1_img
                lang2_subtitle_img = lang2_img

                self.time_loc_list.append(self.get_time_loc(current_frame))

                lang1_last_img = lang1_img
                lang2_last_img = lang2_img

                # print("1-1", lang1_img.size)
                # print("1-2", lang1_subtitle_img.size)
                # print()

                current_row_num += 1
            else:
                if std_err(lang1_img, lang1_last_img) > 1 and std_err(lang2_img, lang2_last_img) > 1:
                    lang1_subtitle_img = np.vstack((lang1_subtitle_img, lang1_img))
                    lang2_subtitle_img = np.vstack((lang2_subtitle_img, lang2_img))

                    self.time_loc_list.append(self.get_time_loc(current_frame))

                    lang1_last_img = lang1_img
                    lang2_last_img = lang2_img

                    # print("2-1", lang1_img.size)
                    # print("2-2", lang1_subtitle_img.size)
                    # print()

                    current_row_num += 1

            # generates a subtitle img
            if current_row_num == MP4Slice.subtitle_row_num or done:
                # Image.fromarray(lang1_subtitle_img).show()
                # Image.fromarray(lang2_subtitle_img).show()
                # input()

                self.lang1_subtitle_img_list.append(lang1_subtitle_img)
                self.lang2_subtitle_img_list.append(lang2_subtitle_img)

                # print(self.time_loc_list)
                # input()

                lang1_subtitle_img = None
                lang2_subtitle_img = None

                current_row_num = 0

        # print(len(subtitle_img_list))

    def get_subtitle_text(self, app_id, api_key, secret_key):
        def compress(nparray_img):
            # nparray img -> pil img
            pil_img = Image.fromarray(nparray_img)

            # resizes the image
            width, height = pil_img.size
            while width * height > 4_000_000:
                width //= 2
                height //= 2

            bin_img = pil_img.resize((width, height), Image.BILINEAR)

            with BytesIO() as f:
                bin_img.save(f, format="PNG")
                # f.seek(0)
                return f.getvalue()

            # return bin_img

        def bin_img_2_subtitle_text(bin_img, language_type, subtitle_text_list):
            client = AipOcr(appId=app_id, apiKey=api_key, secretKey=secret_key)

            # specifies the language type
            options = {"language_type": language_type}

            try:
                message = client.basicAccurate(bin_img, options=options)
                for text in message.get("words_result"):
                    # print(text.get("words"))
                    subtitle_text_list.append(text.get("words"))
            except:
                pass

        def get_subtitle_text_main(subtitle_img_list, subtitle_text_list, language_type):
            for subtitle_img in subtitle_img_list:
                # compresses the img
                bin_img = compress(subtitle_img)

                # gets subtitles
                bin_img_2_subtitle_text(bin_img, language_type, subtitle_text_list)

        self.lang1_subtitle_text_list = []
        self.lang2_subtitle_text_list = []
        get_subtitle_text_main(self.lang1_subtitle_img_list, self.lang1_subtitle_text_list, self.lang1)
        get_subtitle_text_main(self.lang2_subtitle_img_list, self.lang2_subtitle_text_list, self.lang2)
        # print(self.lang1_subtitle_text_list)
        # print(self.lang2_subtitle_text_list)


class Producer(threading.Thread):
    def __init__(self, thread_name, mp4, mp4_slice_queue, app_id, api_key, secret_key):
        super(Producer, self).__init__()
        self.thread_name = thread_name
        self.mp4 = mp4
        self.mp4_slice_queue = mp4_slice_queue
        self.app_id = app_id
        self.api_key = api_key
        self.secret_key = secret_key

        self.count = 0
        self.max_count = len(self.mp4.frame_slice_list)

    def run(self):
        while self.count < self.max_count:
            frame_slice = self.mp4.frame_slice_list[0]

            # creates a mp4 slice obj
            mp4_slice = MP4Slice(self.mp4, frame_slice=frame_slice)

            mp4_slice.get_subtitle_img()
            mp4_slice.get_subtitle_text(self.app_id, self.api_key, self.secret_key)

            self.mp4_slice_queue.put(mp4_slice)

            global producer_count_lock
            try:
                producer_count_lock.acquire()
                self.count += 1
            finally:
                producer_count_lock.release()


class Consumer(threading.Thread):
    def __init__(self, thread_name, mp4, mp4_slice_queue, store_dir):
        super(Consumer, self).__init__()
        self.thread_name = thread_name
        self.mp4 = mp4
        self.mp4_slice_queue = mp4_slice_queue
        self.store_dir = store_dir

        self.count = 0
        self.max_count = len(mp4.frame_slice_list)

    def run(self):
        while self.count < self.max_count:
            mp4_slice = self.mp4_slice_queue.get()

            # saves (subtitle_img, audio_slice, subtitle_text) in a dir
            self.save_data(mp4_slice)

            global consumer_count_lock
            try:
                consumer_count_lock.acquire()
                self.count += 1
            finally:
                consumer_count_lock.release()

    def save_data(self, mp4_slice):
        def save_subtitle_img(subtitle_img_list):
            for i in range(len(subtitle_img_list)):
                img_content = subtitle_img_list[i]
                img_path = os.path.join(dir_path, f"{i}.png")
                cv2.imwrite(img_path, img_content)

        def save_audio_piece():
            subprocess.call(
                f"ffmpeg -ss {mp4_slice.time_slice_tuple[0]} -t {self.mp4.time_slice_len} -i {self.mp4.audio_path} "
                f"-c copy {os.path.join(dir_path, time_slice + '.mp3')} &> /dev/null",
                shell=True)

        def save_subtitle_text(subtitle_text_list, lang_type):
            subtitle_text_file = os.path.join(dir_path, f"{lang_type}_{time_slice}.txt")
            with open(subtitle_text_file, "w") as f:
                for subtitle_text in subtitle_text_list:
                    f.write(subtitle_text + "\t" * 6 + "\n")

        # generates a dir for storing the slice
        time_slice = f"{mp4_slice.time_slice_tuple[0]}-{mp4_slice.time_slice_tuple[1]}"
        dir_path = os.path.join(self.store_dir, time_slice)
        make_dir(dir_path)

        # saves the subtitle img
        save_subtitle_img(mp4_slice.lang1_subtitle_img_list)
        save_subtitle_img(mp4_slice.lang2_subtitle_img_list)

        # generates the audio slice and saves it
        save_audio_piece()

        # saves the subtitle text
        save_subtitle_text(mp4_slice.lang1_subtitle_text_list, self.mp4.lang1)
        save_subtitle_text(mp4_slice.lang2_subtitle_text_list, self.mp4.lang2)


def mp4_2_mp3(mp4):
    mp4.audio_path = os.path.splitext(mp4.video_path)[0] + '.mp3'
    if not os.path.exists(mp4.audio_path):
        subprocess.call(f"ffmpeg -i {mp4.video_path} {mp4.audio_path} &> /dev/null", shell=True)


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except:
        pass


def create_threads(mp4, mp4_slice_queue, store_dir, app_id, api_key, secret_key):
    producer = Producer("provider", mp4, mp4_slice_queue, app_id, api_key, secret_key)
    consumer = Consumer("consumer", mp4, mp4_slice_queue, store_dir)

    return producer, consumer


def start_n_join_threads(producer, consumer):
    producer.start()
    consumer.start()

    producer.join()
    consumer.join()


def subtitle_recognition(video_path, time_slice_len, lang1, lang2, store_dir, app_id, api_key, secret_key):
    # reads the video
    video_cap = cv2.VideoCapture(video_path)

    # creates an MP4 obj
    mp4 = MP4()
    mp4.video_path = video_path
    mp4.fps = video_cap.get(cv2.CAP_PROP_FPS)
    mp4.frame_total_num = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mp4.get_length()
    mp4.time_slice_len = time_slice_len
    mp4.get_frame_slice_list()
    mp4.lang1 = lang1
    mp4.lang2 = lang2
    mp4.lang1_subtitle_loc = [620, 670]
    mp4.lang2_subtitle_loc = [670, 710]
    mp4.video_cap = video_cap

    # mp4 -> mp3 to get the audio
    mp4_2_mp3(mp4)

    # print(mp4.video_path)
    # print(mp4.fps)
    # print(mp4.frame_total_num)
    # print(mp4.length)
    # print(mp4.time_slice_len)
    # print(mp4.frame_slice_list)
    # print(mp4.lang1)
    # print(mp4.lang2)
    # print(mp4.lang1_subtitle_loc)
    # print(mp4.lang2_subtitle_loc)
    # print(mp4.audio_path)
    # print(mp4.video_cap)
    # input()

    # makes a dir for storing data
    make_dir(store_dir)

    # creates a queue for storing mp4_slice objs containing (subtitle img, audio slice, subtitle text)
    mp4_slice_queue = queue.Queue()

    # creates a thread for getting (subtitle img, audio slice, subtitle text)
    # and a thread for saving them
    producer, consumer = create_threads(mp4, mp4_slice_queue, store_dir, app_id, api_key, secret_key)

    # starts and joins the threads
    start_n_join_threads(producer, consumer)


if __name__ == '__main__':
    producer_count_lock = threading.Lock()
    consumer_count_lock = threading.Lock()

    app_id = ""
    api_key = ""
    secret_key = ""
    

    subtitle_recognition("test1.mp4", 60, "ZHN_ENG", "JAP", os.getcwd() + "test1", app_id, api_key, secret_key)
