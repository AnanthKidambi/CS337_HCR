# Neural Style Transfer

This repository is an implementation of the research papers "[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)" and "[Artistic style transfer for videos](https://arxiv.org/abs/1604.08610)", done as part of the CS337 project.

## Running Instructions

- Create a virtual environment using `python -m venv .`
- Run `source bin/activate`
- Install dependencies using `pip install -r requirements.txt`
- For video style transfer, put the video in `input/content/` and the style image in `input/style/`, add entries to the dictionary `abbrev_to_full` in `utils.py`.
- Run `python runnner.lua --videoname <video name> --stylename <stylename> --n_threads <number of parallel threads>`
- For image style transfer, update the path of the content and style images in `neural_style.py`, and then run `python neural_style.py`
