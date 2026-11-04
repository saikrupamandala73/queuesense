# QueueSense

**Video-based adaptive traffic signal timing prototype**

QueueSense estimates traffic demand from an intersection video and recommends how to divide green time between north/south and east/west approaches. It is built with Python, Streamlit, OpenCV, and a pretrained YOLOv8 nano detector.

## The problem

Many intersections use fixed signal timings even when traffic demand changes by direction. A fixed split can give equal green time to a quiet approach and a busier one. QueueSense demonstrates a simple way to use camera detections to recommend a more responsive split.

## How QueueSense works

1. The user uploads a short intersection video.
2. YOLOv8 nano detects cars, motorcycles, buses, and trucks in sampled frames.
3. The app places detections into four image regions representing north, south, east, and west approaches.
4. North+south and east+west counts are smoothed over time to reduce sudden changes from missed or noisy detections.
5. A bounded controller allocates green time in proportion to the smoothed counts, while respecting minimum and maximum green limits.
6. The app displays the annotated frame, queue estimates, changing green split, comparison with equal fixed timing, and a downloadable CSV.

## What is included

- `app.py` — Streamlit application and adaptive timing logic.
- `requirements.txt` — Python package dependencies, including CPU-only PyTorch.
- `.gitignore` — files that should stay out of the public repository.

## Run locally

Requires Python 3.10 or newer.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`. On first run, Ultralytics downloads the YOLOv8 nano weights if they are not already available. CPU inference is supported; a GPU is not required.

## Demo results

For the sample clip shown in the project demo, the app reported average estimated detections of **3.8 N/S** and **2.5 E/W**, with a typical recommended split of about **22s / 20s**. These are outputs from one video and one simple region layout; they are not measured traffic counts or proof of reduced waiting time.

Add screenshots or a short demo recording here:

- `![QueueSense dashboard](screenshots/queuesense-results.png)`
- Demo video: TODO

## Limitations and next steps

- The app replays an uploaded clip; it does not connect to a live city camera.
- Counts are detections in image regions, not tracked unique vehicles or validated queue lengths.
- The four regions are simple fixed image areas and need tuning for each camera angle.
- Green-time recommendations are a prototype policy. The app does not control real signals, simulate vehicle movement, or measure actual delay reduction.

Possible next steps include vehicle tracking, user-adjustable approach regions, a traffic simulator for before/after evaluation, and a live camera input.

## Tools used

- **Python, Streamlit, OpenCV, Ultralytics YOLOv8 nano, NumPy, and pandas** — application and video-analysis stack.

