# QueueSense — FT-01 Smart Traffic Signal Optimisation

QueueSense is a Python prototype that analyzes a local intersection video with YOLOv8 nano and updates a recommended N/S versus E/W green-time split as sampled frames are processed. It is designed as a demonstrable edge-AI concept for Neural Storm FT-01.

## What the demo does

- Runs a pretrained YOLOv8 nano detector locally on the uploaded clip.
- Counts cars, motorcycles, buses, and trucks inside four simple approach regions.
- Smooths frame-to-frame count changes with an exponential moving average.
- Recommends an adaptive split within minimum and maximum green limits, preserving a feasible cycle.
- Shows queue-estimate and controller-response charts, a fixed equal-split baseline, and downloadable frame-level CSV results.

The adaptive recommendation changes as sampled frames are analyzed. The current app replays an uploaded clip; it does not connect to a live intersection or operate a real signal.

## Run locally

Requires Python 3.10 or newer.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. On first run, Ultralytics downloads `yolov8n.pt` if it is not already present. CPU inference works but may be slower; a GPU is optional.

## Controller logic

The video frame is passed to the pretrained detector. Vehicle box centers are assigned to four fixed image regions (N, S, E, W). The paired estimates form N/S and E/W demand. An exponential moving average reduces short frame-to-frame detection jitter. The adaptive controller divides a cycle between the two approaches in proportion to their smoothed estimates, subject to configurable green limits. The baseline uses an equal split.

The region geometry in `approach_counts()` is intentionally simple and should be tuned to the chosen camera view. Use a stable elevated clip that shows all four approaches for the most understandable demonstration.

## How to describe results accurately

The displayed values are **vehicle detections in image regions**, not tracked unique vehicles or validated queue lengths. The app recommends signal timing but does not simulate arrivals and departures, measure actual waiting time, or establish congestion reduction. Do not claim a measured improvement from the charts. A rigorous next step would add vehicle tracking and compare both controllers in a traffic simulator or controlled field study.

## Project demo checklist

1. Use a short four-way intersection clip and tune the regions if needed.
2. Run the app and capture the summary, queue chart, controller-response chart, and comparison table.
3. Download the frame-level CSV and describe it as model output, not ground truth.
4. Add a short screen recording or GIF showing the recommendation updating while frames are processed.

## Hackathon submission checklist

- Make the GitHub repository public.
- Include `app.py`, `requirements.txt`, `.gitignore`, this README, and a results screenshot.
- Fill in the team name and member names below.
- Disclose AI development tools used, including ChatGPT or GitHub Copilot, if applicable.
- Add a demo/deployment link if available and verify that the repository is accessible.

## Team

- Team name: TODO
- Members: TODO

## AI tool disclosure

TODO: List tools used and briefly describe their role in the project.
