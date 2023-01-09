# Subtitle Recognition

A tool for recognizing multilingual subtitles from videos.  
从视频提取多语字幕的工具。

## Dependencies · 依赖
```bash
pip install -r requirements.txt
```

## Usage · 使用方法
```bash
python3.7 main.py \
  --video-path test.mp4 \
  --lang CHN JAP \
  --app-id APP_ID \
  --api-key API_KEY \
  --secret-key SECRET_KEY
```

The `--lang` argument specifies the languages of the subtitle. `--app-id`, `--api-key`, `--secret-key` are for OCR and can be obtained from [Baidu AI Cloud](https://cloud.baidu.com/). "通用文字识别（标准版）" is used.  
`--lang` 参数指定字幕的语言。`--app-id`, `api-key`, `--secret-key`用于文字识别，可以从[百度智能云](https://cloud.baidu.com/)获取。代码使用的是"通用文字识别（标准版）"。