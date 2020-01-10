#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import queue
import threading
import datetime
from io import BytesIO
import re
import argparse
import random

import cv2
import numpy as np
from PIL import Image
from aip import AipOcr


class MP4:
    def __init__(self, video_path="", video_cap=None, audio_path="", fps=0, frame_total_num=0, length=0,
                 time_slice_len=0, frame_slice_pair_list=None, lang1="", lang2="", lang1_subtitle_loc=None,
                 lang2_subtitle_loc=None):
        self.video_path = video_path
        self.video_cap = video_cap
        self.audio_path = audio_path
        self.fps = fps
        self.frame_total_num = frame_total_num
        self.length = length  # represented in secs
        self.time_slice_len = time_slice_len
        self.frame_slice_pair_list = frame_slice_pair_list  # each elem: (frame slice start num, frame slice end num)
        self.lang1 = lang1
        self.lang2 = lang2
        self.lang1_subtitle_loc = lang1_subtitle_loc
        self.lang2_subtitle_loc = lang2_subtitle_loc

    def set_length(self):
        self.length = self.frame_total_num / self.fps

    def set_frame_slice_pair_list(self):
        self.frame_slice_pair_list = []

        current_sec = 0  # the time is represented in secs here
        while current_sec < self.length:
            frame_slice_start_num = int(current_sec * self.fps)
            current_sec += self.time_slice_len
            frame_slice_end_num = int(current_sec * self.fps)

            self.frame_slice_pair_list.append([frame_slice_start_num, frame_slice_end_num])

        # frame_slice_end_num in the last pair calculated may be longer than the actual total video length
        self.frame_slice_pair_list[-1][1] = self.frame_total_num - 1

    def set_subtitle_loc(self):
        def on_event_left_button(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:  # if left clicked
                # puts the y-coordinate to the list
                loc.append(y)

                # visualizes the action of the mouse
                cv2.circle(img=img_cp, center=(x, y), radius=2, color=(255, 0, 0), thickness=-1)
                cv2.putText(img=img_cp, text=f"x: {x}, y: {y}", org=(x, y), fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=1.0,
                            color=(255, 0, 0), thickness=1)
                cv2.imshow("frame", img_cp)

        def get_loc(lang, lang_subtitle_loc, img):
            print(f"click the top and bottom of the {lang} subtitle. "
                  f"press Enter to finish selecting the location of {lang} subtitle, press ESC to select again")
            while True:
                nonlocal loc
                nonlocal img_cp

                cv2.setMouseCallback("frame", on_event_left_button)
                key = cv2.waitKey(15 * 1_000)
                try_open()

                if len(loc) == 2:
                    lang_subtitle_loc[0] = loc[0]
                    lang_subtitle_loc[1] = loc[1]
                    print(f"the top and bottom of the subtitle of {lang}: "
                          f"{lang_subtitle_loc[0]}, {lang_subtitle_loc[1]}")

                    while key not in [13, 27]:
                        print("press ESC or Enter only")
                        key = cv2.waitKey(15 * 1_000)
                        try_open()

                    if key == 13:
                        img_cp = img.copy()
                        cv2.imshow("frame", img_cp)

                        return

                loc = []
                print(f"click two points in the frame to specify the top and bottom of the {lang} subtitle")

                img_cp = img.copy()
                cv2.imshow("frame", img_cp)

        def try_open():
            if cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) == 0:
                cv2.imshow("frame", img_cp)

        self.lang1_subtitle_loc = [None, None]
        self.lang2_subtitle_loc = [None, None]

        cv2.namedWindow("frame")

        # uses the middle frame in the video as the init frame to be displayed
        frame_num = self.frame_total_num // 2

        while True:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            _, img = self.video_cap.read()
            img_cp = img.copy()

            print("press ESC if there is no title in the frame, else press Enter")

            cv2.imshow("frame", img_cp)
            key = cv2.waitKey(15 * 1_000)
            try_open()

            while key not in [13, 27]:
                print("press Enter or ESC only")
                key = cv2.waitKey(15 * 1_000)
                try_open()

            if key == 13:
                loc = []
                get_loc(self.lang1, self.lang1_subtitle_loc, img)

                loc = []
                get_loc(self.lang2, self.lang2_subtitle_loc, img)

                cv2.destroyWindow("frame")

                break
            else:
                frame_num = random.randint(0, self.frame_total_num)


class MP4Slice(MP4):
    subtitle_row_num = 6  # num of stacked subtitles in images generated for ocr

    def __init__(self, mp4, frame_slice_pair=None, lang1_subtitle_img_list=None, lang2_subtitle_img_list=None,
                 lang1_subtitle_text_list=None, lang2_subtitle_text_list=None):
        super(MP4Slice, self).__init__(mp4.video_path, mp4.video_cap, mp4.audio_path, mp4.fps, mp4.frame_total_num,
                                       mp4.length, mp4.time_slice_len, mp4.frame_slice_pair_list, mp4.lang1, mp4.lang2,
                                       mp4.lang1_subtitle_loc, mp4.lang2_subtitle_loc)
        self.frame_slice_pair = frame_slice_pair  # start frame num, end frame num
        self.time_slice_pair = (self.get_time_loc(self.frame_slice_pair[0]),
                                self.get_time_loc(self.frame_slice_pair[1]))  # start time slice, end time slice
        self.lang1_subtitle_img_list = lang1_subtitle_img_list
        self.lang2_subtitle_img_list = lang2_subtitle_img_list
        self.lang1_subtitle_text_list = lang1_subtitle_text_list
        self.lang2_subtitle_text_list = lang2_subtitle_text_list

    def get_time_loc(self, current_frame):
        total_sec = int(current_frame / self.fps)

        hour, minute, second = sec_2_hms(total_sec)

        return datetime.time(hour=hour, minute=minute, second=second).isoformat()

    def retrieve_subtitle_img(self):
        def std_err(img_current_frame, img_last_frame=None):
            if img_last_frame is None:  # checks if the img contains a subtitle
                return (img_current_frame ** 2).sum() / img_current_frame.size * 100
            else:  # checks if the subtitle in the current frame is the same as the one in the last frame
                return ((img_current_frame - img_last_frame) ** 2).sum() / img_current_frame.size * 100

        print(f"retrieving subtitles as images from {self.time_slice_pair[0]} to {self.time_slice_pair[1]}")

        # gets a frame slice
        self.frame_slice_pair = self.frame_slice_pair_list.pop(0)

        lang1_last_img = None
        lang2_last_img = None
        lang1_subtitle_img = None
        lang2_subtitle_img = None
        self.lang1_subtitle_img_list = []
        self.lang2_subtitle_img_list = []
        current_row_num = 0

        current_frame = self.frame_slice_pair[0]
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        while current_frame <= self.frame_slice_pair[1]:
            # reads a frame
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

            # stacks the subtitles with the specified number of rows per image
            if current_row_num == 0:
                lang1_subtitle_img = lang1_img
                lang2_subtitle_img = lang2_img

                lang1_last_img = lang1_img
                lang2_last_img = lang2_img

                current_row_num += 1
            else:
                # if the subtitle in the current frame is not the same as the subtitle in the last frame
                if std_err(lang1_img, lang1_last_img) > 1 and std_err(lang2_img, lang2_last_img) > 1:
                    lang1_subtitle_img = np.vstack((lang1_subtitle_img, lang1_img))
                    lang2_subtitle_img = np.vstack((lang2_subtitle_img, lang2_img))

                    lang1_last_img = lang1_img
                    lang2_last_img = lang2_img

                    current_row_num += 1

            # generates a subtitle img
            if current_row_num == MP4Slice.subtitle_row_num or current_frame == self.frame_slice_pair[1]:
                self.lang1_subtitle_img_list.append(lang1_subtitle_img)
                self.lang2_subtitle_img_list.append(lang2_subtitle_img)

                lang1_subtitle_img = None
                lang2_subtitle_img = None

                current_row_num = 0

            # jumps
            for _ in range(9):
                self.video_cap.read()
                current_frame += 1

    def retrieve_subtitle_text(self, app_id, api_key, secret_key):
        def compress(nparray_img):
            # nparray img -> pil img
            pil_img = Image.fromarray(nparray_img)

            # resizes the image til its size is less than 4M,
            # for that baidu ocr receives imgs whose sizes are below 4M
            width, height = pil_img.size
            while width * height > 4_000_000:
                width //= 2
                height //= 2

            pil_img = pil_img.resize((width, height), Image.BILINEAR)

            # pil img -> bin img with a png format
            with BytesIO() as f:
                pil_img.save(f, format="PNG")

                return f.getvalue()

        def remove_punctuations(text):
            pat = re.compile(r"\W")
            text = pat.sub('', text)

            return text

        def bin_img_2_subtitle_text(bin_img, language_type, subtitle_text_list):
            client = AipOcr(appId=app_id, apiKey=api_key, secretKey=secret_key)

            # specifies the language type
            if self.lang1 in ["CHN", "ENG"]:
                self.lang1 = "CHN_ENG"
            if self.lang2 in ["CHN", "ENG"]:
                self.lang2 = "CHN_ENG"

            options = {"language_type": language_type}

            message = {}
            # ocr
            try:
                message = client.basicAccurate(bin_img, options=options)
                if "error_code" in message.keys() and message["error_code"] == 17:
                    # open api daily request limit reached
                    message = client.basicGeneral(bin_img, options=options)
                words_result = message.get("words_result")

                # puts subtitle texts into the list
                for text in words_result:
                    ocr_text = text.get("words")
                    # removes repeated subtitles
                    if len(subtitle_text_list) == 0 or remove_punctuations(ocr_text) != remove_punctuations(
                            subtitle_text_list[-1]):
                        subtitle_text_list.append(ocr_text)
            except TypeError:
                print(f"{message['error_code']}: {message['error_msg']}")
            except:
                print(message)

        def retrieve_subtitle_text_main(subtitle_img_list, subtitle_text_list, language_type):
            for subtitle_img in subtitle_img_list:
                # compresses the img
                bin_img = compress(subtitle_img)

                # gets subtitles
                bin_img_2_subtitle_text(bin_img, language_type, subtitle_text_list)

        print(f"converting subtitle images into texts from {self.time_slice_pair[0]} to {self.time_slice_pair[1]}")

        self.lang1_subtitle_text_list = []
        self.lang2_subtitle_text_list = []
        retrieve_subtitle_text_main(self.lang1_subtitle_img_list, self.lang1_subtitle_text_list, self.lang1)
        retrieve_subtitle_text_main(self.lang2_subtitle_img_list, self.lang2_subtitle_text_list, self.lang2)


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
        self.max_count = len(self.mp4.frame_slice_pair_list)

    def run(self):
        while self.count < self.max_count:
            frame_slice_pair = self.mp4.frame_slice_pair_list[0]

            # creates a mp4 slice obj
            mp4_slice = MP4Slice(self.mp4, frame_slice_pair=frame_slice_pair)

            # gets subtitle images
            mp4_slice.retrieve_subtitle_img()

            # gets subtitle texts
            mp4_slice.retrieve_subtitle_text(self.app_id, self.api_key, self.secret_key)

            # puts the mp4slice obj into the queue
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
        self.max_count = len(mp4.frame_slice_pair_list)

    def run(self):
        while self.count < self.max_count:
            mp4_slice = self.mp4_slice_queue.get()

            # saves subtitle_img, audio_slice, subtitle_text in a dir
            self.save_data(mp4_slice)

            global consumer_count_lock
            try:
                consumer_count_lock.acquire()
                self.count += 1
            finally:
                consumer_count_lock.release()

    def save_data(self, mp4_slice):
        def save_subtitle_img(subtitle_img_list, lang):
            print(f"saving subtitle images from {mp4_slice.time_slice_pair[0]} to {mp4_slice.time_slice_pair[1]}")

            for i in range(len(subtitle_img_list)):
                subtitle_img_content = subtitle_img_list[i]
                subtitle_img_path = os.path.join(dir_path, f"{lang}_{time_slice_str}_{i}.png")
                cv2.imwrite(subtitle_img_path, subtitle_img_content)

        def save_subtitle_text(subtitle_text_list, lang):
            print(f"saving subtitle texts from {mp4_slice.time_slice_pair[0]} to {mp4_slice.time_slice_pair[1]}")

            # removes remaining repeated subtitles
            subtitle_text_list_ = list(set(subtitle_text_list))
            subtitle_text_list_.sort(key=subtitle_text_list.index)

            subtitle_txt_path = os.path.join(dir_path, f"{lang}_{time_slice_str}.txt")
            with open(subtitle_txt_path, "w") as f:
                for subtitle_text in subtitle_text_list_:
                    f.write(subtitle_text + "\n")

        def save_audio_slice():
            print(f"generating audio slice from {mp4_slice.time_slice_pair[0]} to {mp4_slice.time_slice_pair[1]}")

            subprocess.call(
                f"ffmpeg -ss {mp4_slice.time_slice_pair[0]} -t {self.mp4.time_slice_len} -i {self.mp4.audio_path} "
                f"-c copy {os.path.join(dir_path, time_slice_str + '.mp3')} &> /dev/null",
                shell=True)

        # generates a dir for storing the slice
        time_slice_str = f"{mp4_slice.time_slice_pair[0]}-{mp4_slice.time_slice_pair[1]}"
        dir_path = os.path.join(self.store_dir, time_slice_str)
        make_dir(dir_path)

        # saves the subtitle img
        save_subtitle_img(mp4_slice.lang1_subtitle_img_list, self.mp4.lang1)
        save_subtitle_img(mp4_slice.lang2_subtitle_img_list, self.mp4.lang2)

        # saves the subtitle text
        save_subtitle_text(mp4_slice.lang1_subtitle_text_list, self.mp4.lang1)
        save_subtitle_text(mp4_slice.lang2_subtitle_text_list, self.mp4.lang2)

        # generates the audio slice and saves it
        save_audio_slice()


def sec_2_hms(total_sec):
    minute, second = divmod(total_sec, 60)
    hour, minute = divmod(minute, 60)

    return hour, minute, second


def mp4_2_mp3(mp4):
    mp4.audio_path = os.path.splitext(mp4.video_path)[0] + '.mp3'

    print(f"converting {os.path.basename(mp4.video_path)} to {os.path.basename(mp4.audio_path)}")

    if not os.path.exists(mp4.audio_path):
        subprocess.call(f"ffmpeg -i {mp4.video_path} {mp4.audio_path} &> /dev/null", shell=True)
    else:
        print(f"{err_msg}corresponding mp3 already exists")


def make_dir(dir_path):
    print(f"trying making directory {dir_path}")

    try:
        os.mkdir(dir_path)
    except FileExistsError:
        print(f"{err_msg}directory already exists")
    except:
        print(f"{err_msg}failed to create {dir_path}")
        exit(1)


def create_threads(mp4, mp4_slice_queue, store_dir, app_id, api_key, secret_key):
    producer = Producer("provider", mp4, mp4_slice_queue, app_id, api_key, secret_key)
    consumer = Consumer("consumer", mp4, mp4_slice_queue, store_dir)

    return producer, consumer


def start_n_join_threads(producer, consumer):
    producer.start()
    consumer.start()

    producer.join()
    consumer.join()


def remove_complete_mp3(mp4):
    print("removing complete mp3")

    os.remove(mp4.audio_path)


def subtitle_recognition(video_path, time_slice_len, lang1, lang2, store_dir, app_id, api_key, secret_key):
    # reads the video
    video_cap = cv2.VideoCapture(video_path)

    # creates an MP4 obj
    mp4 = MP4()
    mp4.video_path = video_path
    mp4.video_cap = video_cap
    mp4.fps = video_cap.get(cv2.CAP_PROP_FPS)
    mp4.frame_total_num = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mp4.set_length()
    mp4.time_slice_len = time_slice_len
    mp4.set_frame_slice_pair_list()
    mp4.lang1 = lang1
    mp4.lang2 = lang2
    mp4.set_subtitle_loc()

    # mp4 -> mp3 to get the audio
    mp4_2_mp3(mp4)

    # makes a dir for storing data
    make_dir(store_dir)

    # creates a queue for storing mp4_slice objs containing subtitle img, audio slice and subtitle text
    mp4_slice_queue = queue.Queue()

    # creates a thread for getting subtitle img, audio slice and subtitle text
    # and a thread for saving them
    producer, consumer = create_threads(mp4, mp4_slice_queue, store_dir, app_id, api_key, secret_key)

    # starts and joins the threads
    start_n_join_threads(producer, consumer)

    # removes the entire mp3 file
    remove_complete_mp3(mp4)


def validate_mp4(input_path):
    if not os.path.exists(input_path):
        raise argparse.ArgumentTypeError("path not exists")
    if not os.path.isfile(input_path) or not os.path.basename(input_path).split('.')[-1] == "mp4":
        raise argparse.ArgumentTypeError("not a mp4 file")

    return input_path


if __name__ == '__main__':
    producer_count_lock = threading.Lock()
    consumer_count_lock = threading.Lock()

    err_msg = "subtitle-recognition.py: error: "

    parser = argparse.ArgumentParser(description="subtitle_recognition.py - a tool for recognizing subtitles in videos"
                                                 " and converting them into texts")

    parser.add_argument("--video-path", "-v", action="store", required=True, type=validate_mp4,
                        help="path of the video")
    parser.add_argument("--time-slice-len", "-t", action="store", default=60, type=int,
                        help="length of the time slice (sec), 60 by default")
    parser.add_argument("--lang", "-l", action="store", required=True, nargs=2,
                        choices=["CHN", "ENG", "POR", "FRE", "GER", "ITA", "SPA", "RUS", "JAP", "KOR"],
                        help="languages of the subtitle")
    parser.add_argument("--store-dir", "-d", action="store", default=os.getcwd(), help="directory for saving data")
    parser.add_argument("--app-id", action="store", required=True, help="app id of baidu ocr")
    parser.add_argument("--api-key", action="store", required=True, help="api key id of baidu ocr")
    parser.add_argument("--secret-key", action="store", required=True, help="secret key id of baidu ocr")

    args = parser.parse_args()

    start_time = datetime.datetime.now()

    subtitle_recognition(args.video_path, args.time_slice_len, args.lang[0], args.lang[1], args.store_dir, args.app_id,
                         args.api_key, args.secret_key)

    finish_time = datetime.datetime.now()

    hour_cost, minute_cost, second_cost = sec_2_hms((finish_time - start_time).seconds)
    print(f"{hour_cost}h {minute_cost}min {second_cost}sec cost")
