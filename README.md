# Subtitle Recognition

A tool for recognizing bilingual subtitles from videos.  
从视频识别提取双语字幕的工具。

## Dependencies · 依赖
```bash
pip install -r requirements.txt
```

## Usage · 使用方法
```bash
python main.py \
  --video test.mp4 \
  --lang1 JAP \
  --lang2 CHN \
  --start "00:00:00" \
  --end "00:01:00" \
  --app-id APP_ID \
  --api-key API_KEY \
  --secret-key SECRET_KEY
```

The `--lang1` and `--lang2` arguments specify the languages of the subtitle. `--start` and `--end` specify the starting time and ending time to recognize. `--app-id`, `--api-key`, `--secret-key` are for OCR and can be obtained from [Baidu AI Cloud OCR](https://ai.baidu.com/ai-doc/index/OCR). The four versions of "通用文字识别" are used.

`--lang1` 和 `--lang2` 参数指定字幕的两种语言。`--start` 和 `--end` 参数指定字幕识别的视频时间区间。`--app-id`, `api-key`, `--secret-key`用于文字识别，可以从[百度智能云文字识别 OCR](https://ai.baidu.com/ai-doc/index/OCR)获取。代码使用的是"通用文字识别"的四个版本。

## References · 参考
+ [基于图像识别和文字识别用 Python 提取视频字幕](https://blog.csdn.net/XnCSD/article/details/89376477)
+ [Python 利用百度文字识别 API 识别并提取图片中文字](https://blog.csdn.net/XnCSD/article/details/80786793)
+ [利用Python提取视频中的字幕（文字识别）](https://zhuanlan.zhihu.com/p/136264493)

Note that in the mentioned posts, images are binarized before sending to the API. However, I found that directly sending images without binarization produces better recognition performance.

提到的博客发送图片给OCR的API前做了图片二值化。然而，实现的时候发现不做二值化直接扔给API识别效果反而更好。