import argparse
import csv
import datetime
import io
import os
import random

import cv2
import numpy as np
from PIL import Image
from aip import AipOcr


def solve_langs(lang1: str, lang2: str) -> tuple:
    """Convert CHN / ENG to CHN_ENG.

    :param lang1: Language 1.
    :param lang2: Language 2.
    :return: (lang1, lang2).
    """

    chn = "CHN"
    chn_eng = "CHN_ENG"

    if lang1 == chn:
        lang1 = chn_eng
    if lang2 in chn:
        lang2 = chn_eng

    return lang1, lang2


def get_bounds(video_cap, n_frames: int, lang1: str, lang2: str) -> tuple:
    """Get the upper and lower bounds of the subtitles in lang1 and lang2.

    :param video_cap: Video to process.
    :param n_frames: Total frames.
    :param lang1: Language 1.
    :param lang2: Language 2.
    :return: (
        lang1_upper, lang1_lower,
        lang2_upper, lang2_lower
    )
    """

    # Enter key.
    enter_key = 13
    # ESC key.
    esc_key = 27
    # Space key.
    space_key = 32

    # Waiting time (milli seconds).
    waiting_time = 30 * 1_000

    def display_frame():
        """Display a frame with subtitles."""

        nonlocal img
        nonlocal img_cp

        while True:
            # Randomly pick a frame to display.
            curr_frame = random.randint(0, n_frames)

            # Display the picked frame.
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
            _, img = video_cap.read()
            img_cp = img.copy()

            # Check if the frame contains subtitles.
            print("Press <Enter> if there are subtitles in the frame, else press <Space>.")
            cv2.imshow(window_name, img_cp)
            key = cv2.waitKey(waiting_time)
            while key not in [enter_key, space_key]:
                print("Press <Enter> or <Space> only.")
                try_open()
                key = cv2.waitKey(waiting_time)

            if key == enter_key:
                break

    def mark(e, x, y, f, p, lang_bounds):
        """Mark the clicked locations."""

        if e == cv2.EVENT_LBUTTONDOWN:
            # Left-clicked.
            # Add the y-coordinate to the list.
            lang_bounds.append(y)

            # Visualize the click.
            cv2.circle(img=img_cp, center=(x, y), radius=2, color=(255, 0, 0), thickness=-1)
            cv2.putText(img=img_cp, text=f"x: {x}, y: {y}", org=(x, y), fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1.0, color=(255, 0, 0), thickness=1)
            cv2.imshow(window_name, img_cp)

    def get(lang):
        nonlocal img_cp

        lang_bounds = []
        while True:
            # Clear the marks if any.
            img_cp = img.copy()
            cv2.imshow(window_name, img_cp)

            print(f"Click the upper and lower bounds of the {lang} subtitle. "
                  f"Press <Enter> to finish locating. Press <ESC> to redo.")
            cv2.setMouseCallback(window_name, lambda e, x, y, f, p: mark(e, x, y, f, p, lang_bounds))
            key = cv2.waitKey(waiting_time)
            try_open()
            while key not in [enter_key, esc_key]:
                print("Press <ESC> or <Enter> only.")
                try_open()
                key = cv2.waitKey(waiting_time)

            if key == enter_key and len(lang_bounds) == 2 and all(lang_bounds):
                print(f"The upper and lower bounds of the {lang} subtitle: "
                      f"({lang_bounds[0]}, {lang_bounds[1]})")
                return lang_bounds
            else:
                print("Click twice.")

            # Redo.
            lang_bounds = []

    def try_open():
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) == 0:
            cv2.imshow(window_name, img_cp)

    img = None
    img_cp = None

    # Create a window for bounds locating.
    window_name = "GetBounds"
    cv2.namedWindow(window_name)

    # Display a frame with subtitles.
    display_frame()

    # Get bounds for lang1.
    # lang1_upper, lang1_lower.
    lang1_bounds = get(lang1)
    # Get bounds for lang2.
    # lang2_upper, lang2_lower.
    lang2_bounds = get(lang2)

    cv2.destroyWindow(window_name)

    # Reset the frame location.
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return (
        lang1_bounds[0], lang1_bounds[1],
        lang2_bounds[0], lang2_bounds[1]
    )


def isoformat2secs(time_str: str) -> int:
    """Convert ISO format time string to seconds.

    :param time_str: Time string in ISO format. E.g., 00:01:00.
    :return: Time represented in seconds.
    """

    hour, minute, second = time_str.split(":")
    hours = int(hour) * 3600
    minutes = int(minute) * 60
    seconds = int(second)

    return hours + minutes + seconds


def solve_frame(start_timestr: str, end_timestr: str, fps: float, n_frames: int) -> tuple:
    """Determine the start frame and end frame.

    :param start_timestr: Starting time in ISO format.
    :param end_timestr: Ending time in ISO format.
    :param fps: Frame per second.
    :param n_frames: Total frames.
    :return: (start_frame, end_frame).
    """

    if start_timestr is not None and end_timestr is not None:
        # Obtain the starting time and ending time for recognition.
        start_secs = isoformat2secs(time_str=start_timestr)
        end_secs = isoformat2secs(time_str=end_timestr)

        # Convert to frame index.
        # Formula: frame_idx = seconds * fps
        start_frame = int(start_secs * fps)
        end_frame = int(end_secs * fps)
    else:
        start_frame = 0
        end_frame = n_frames

    return start_frame, end_frame


def secs2isoformat(seconds: int) -> str:
    """Format time from seconds to ISO format.

    :param seconds: Second representation of the time.
    :return: Formatted time (HH:MM:SS).
    """

    hours = seconds // 3600
    minutes = (seconds - hours * 3600) // 60
    seconds = seconds - hours * 3600 - minutes * 60

    t = datetime.time(hour=hours, minute=minutes, second=seconds)
    return datetime.time.isoformat(t)


def frame2time(frame_idx: int, fps: int) -> str:
    """Convert the frame at frame_idx to time format.

    :param frame_idx: Idx of the frame.
    :param fps: FPS.
    :return: Formatted time for the frame.
    """

    return secs2isoformat(seconds=int(frame_idx / fps))


def stderr(img1: np.ndarray, img2: np.ndarray = None) -> float:
    """Calculate the standard error of the image(s).

    :param img1: Image 1.
    :param img2: Image 2 (optional).
    :return: Standard error of the image(s).
    """

    if img2 is None:
        # The std err of an image.
        return (img1 ** 2).sum() / img1.size * 100
    else:
        # The std err of two images.
        return ((img1 - img2) ** 2).sum() / img1.size * 100


def has_text(img: np.ndarray) -> bool:
    """Check if the image contains text.

    :param img: Image to check.
    :return: True if the image contains text, else False.

    If the std err of the image > 1, the image
    contains text. Else not.
    """

    return stderr(img1=img) > 1


def is_same(img1: np.ndarray, img2: np.ndarray) -> bool:
    """Check if the two images contain the same text.

    :param img1: Image 1.
    :param img2: Image 2.
    :return: True if the two images contain the same text, else False.

    If the std err of the two images <= 1, the images
    contain the same text. Else not.
    """

    return stderr(img1=img1, img2=img2) <= 1


def compress(img: np.ndarray, threshold: int) -> Image:
    """Compress the image.

    :param img: Image to compress.
    :param threshold: Size threshold (width * height).
    :return: Compressed image.
    """

    # Numpy -> PIL.
    img = Image.fromarray(img)

    width, height = img.size
    while width * height > threshold:
        width = width // 2
        height = height // 2

    new_img = img.resize(
        size=(width, height),
        resample=Image.BILINEAR
    )

    return new_img


def ocr(img: Image, lang: str, api_config: dict) -> list:
    """Recognize text from the image.

    :param img: Image containing text.
    :param lang: Language of the subtitle.
    :param api_config: API config for Baidu OCR (app id, api key, secret key).
    :return: Recognized text.

    Baidu OCR is used for text recognition.
    https://ai.baidu.com/ai-doc/OCR/
    """

    # PIL img -> bin img with a png format.
    with io.BytesIO() as f:
        img.save(f, format="PNG")
        img = f.getvalue()

    ocr_client = AipOcr(
        appId=api_config["app_id"],
        apiKey=api_config["api_key"],
        secretKey=api_config["secret_key"]
    )

    options = {"language_type": lang}
    for ocr_api in [ocr_client.basicAccurate, ocr_client.basicGeneral]:
        rst = ocr_api(
            image=img,
            options=options
        )

        if "error_code" in rst.keys() and rst["error_code"] == 17:
            # Request limit reached.
            continue
        elif "error_code" in rst.keys():
            print(rst)
            raise
        else:
            break

    text = ""
    for word_rst in rst.get("words_result"):
        text = text + word_rst.get("words")

    return text


def main(args: argparse.Namespace):
    # Solve languages.
    lang1, lang2 = solve_langs(
        lang1=args.lang1,
        lang2=args.lang2
    )

    # Read the video
    video_cap = cv2.VideoCapture(args.video)
    fps = video_cap.get(cv2.CAP_PROP_FPS)  # Frame per second.
    n_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Obtain the upper and lower bounds of the subtitles.
    # Note that lang1 and lang2 before solve_langs() are used
    # to get rid of CHN_ENG in prompts.
    (
        lang1_upper, lang1_lower,
        lang2_upper, lang2_lower
    ) = get_bounds(
        video_cap=video_cap,
        n_frames=n_frames,
        lang1=args.lang1,
        lang2=args.lang2,
    )

    # Handle the frame range for recognition.
    start_frame, end_frame = solve_frame(
        start_timestr=args.start,
        end_timestr=args.end,
        fps=fps,
        n_frames=n_frames,
    )
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    curr_frame = start_frame

    api_config = {
        "app_id": args.app_id,
        "api_key": args.api_key,
        "secret_key": args.secret_key
    }

    # Main loop.

    # {
    #   start: start,
    #   lang1: lang1_text,
    #   lang2: lang2_text,
    # }
    subtitles = []

    # Process an image per sec
    # because subtitles are not likely to change in one second.
    n_skip = int(fps)

    # For duplication check.
    last_imgs = {
        lang1: None,
        lang2: None,
    }
    # For duplication check.
    last_subtitles = {
        lang1: None,
        lang2: None
    }

    while True:
        # Skip.
        for _ in range(n_skip):
            video_cap.read()
            curr_frame += 1

        # Read a frame.
        is_success, frame = video_cap.read()
        curr_frame += 1
        if not is_success:
            # Failed to read an image.
            print(f"Failed to read the frame at {frame2time(frame_idx=curr_frame, fps=fps)}.")
            continue
        if frame is None or curr_frame > end_frame:
            # Finished processing the entire video.
            print("Finished processing.")
            break

        # Convert the frame to an image.
        img = frame[:, :, 0]

        # Crop for the subtitle area.
        lang1_img = img[lang1_upper:lang1_lower, :]
        lang2_img = img[lang2_upper:lang2_lower, :]

        # Binarization.
        _, lang1_img = cv2.threshold(lang1_img, args.bin_thred, 255, cv2.THRESH_BINARY)
        _, lang2_img = cv2.threshold(lang2_img, args.bin_thred, 255, cv2.THRESH_BINARY)

        if not has_text(img=lang1_img) or not has_text(img=lang2_img):
            continue
        if is_same(img1=lang1_img, img2=last_imgs[lang1]) or is_same(img1=lang2_img, img2=last_imgs[lang2]):
            continue
        print(f"Processing {frame2time(frame_idx=curr_frame, fps=fps)}")

        # Compress the image.
        lang1_img = compress(img=lang1_img, threshold=args.cmp_thred)
        lang2_img = compress(img=lang2_img, threshold=args.cmp_thred)

        # Recognize text in the image with Baidu OCR.
        lang1_text = ocr(img=lang1_img, lang=lang1, api_config=api_config)
        lang2_text = ocr(img=lang2_img, lang=lang2, api_config=api_config)
        # Duplication check.
        if lang1_text == last_subtitles[lang1] or lang2_text == last_subtitles[lang2]:
            continue
        # Empty check.
        if len(lang1_text) == 0 or len(lang2_text) == 0:
            continue

        subtitles.append({
            "time": frame2time(frame_idx=curr_frame, fps=fps),
            lang1: lang1_text,
            lang2: lang2_text,
        })

        last_imgs[lang1] = lang1_img
        last_imgs[lang2] = lang2_img

        last_subtitles[lang1] = lang1_text
        last_subtitles[lang2] = lang2_text

    video_name = os.path.splitext(args.video)[0]
    fp_txt = f"{video_name}.txt"
    with open(fp_txt, "w", encoding="utf-8") as f:
        tsv_writer = csv.writer(f, delimiter="\t")
        tsv_writer.writerow(["time", lang1, lang2])
        for subtitle in subtitles:
            tsv_writer.writerow([
                subtitle["time"], subtitle[lang1], subtitle[lang2]
            ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video", required=True,
        help="Video containing subtitles."
    )
    parser.add_argument(
        "--lang1", required=True, type=str,
        choices=["CHN", "ENG", "POR", "FRE", "GER", "ITA", "SPA", "RUS", "JAP", "KOR"],
        help="Language 1 of the subtitle."
    )
    parser.add_argument(
        "--lang2", required=True, type=str,
        choices=["CHN", "ENG", "POR", "FRE", "GER", "ITA", "SPA", "RUS", "JAP", "KOR"],
        help="Language 2 of the subtitle."
    )
    parser.add_argument(
        "--start", type=str,
        help="Starting time for recognition. E.g., 00:00:00."
    )
    parser.add_argument(
        "--end", type=str,
        help="Ending time for recognition. E.g., 00:10:00."
    )

    # Hyper params.
    parser.add_argument(
        "--bin-thred", default=220, type=float,
        help="Threshold for binarizing images. 220 by default."
    )
    parser.add_argument(
        "--cmp-thred", default=4_000_000, type=int,
        help="Threshold for compressing images. 200+k by default."
    )

    # API config.
    parser.add_argument(
        "--app-id", required=True, type=str,
        help="App ID of Baidu OCR."
    )
    parser.add_argument(
        "--api-key", required=True, type=str,
        help="Api key of Baidu OCR."
    )
    parser.add_argument(
        "--secret-key", required=True, type=str,
        help="Secret key of Baidu OCR."
    )

    args = parser.parse_args()

    app_id = "29614200"
    api_key = "VKLCRmKbnl0zfCqPSeAV8VcG"
    secret_key = "69vA3P7NrIfQECB5FDDzDoIOHZ8r1r5z"

    main(args)
