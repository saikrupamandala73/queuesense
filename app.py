import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="QueueSense", page_icon="🚦", layout="wide")
st.title("🚦 QueueSense | Adaptive Signal Control")
st.caption("Local edge-AI traffic analysis · FT-01 Smart Traffic Signal Optimisation")
st.info(
    "Prototype only: the detector analyzes a local video and recommends signal timing. "
    "It does not control real signals or establish real-world congestion reduction."
)

with st.sidebar:
    st.header("Controller settings")
    fixed_green = st.slider("Fixed baseline green per phase (s)", 10, 60, 30, 5)
    min_green = st.slider("Adaptive minimum green (s)", 10, 30, 15, 5)
    max_green = st.slider("Adaptive maximum green (s)", 30, 90, 60, 5)
    frame_step = st.slider("Analyze every Nth frame", 1, 15, 5)
    confidence = st.slider("YOLO confidence", 0.10, 0.90, 0.35, 0.05)
    st.caption("Local YOLOv8n inference. The video is processed on this machine.")

uploaded = st.file_uploader("Upload a short, stable-view intersection video", type=["mp4", "mov", "avi", "mkv"])


@st.cache_resource(show_spinner="Loading local YOLOv8 nano…")
def load_model():
    from ultralytics import YOLO

    weights = Path(__file__).with_name("yolov8n.pt")
    return YOLO(str(weights) if weights.exists() else "yolov8n.pt")


def approach_counts(result, width, height):
    """Count vehicles in four simple outer approach ROIs; tune for the camera view."""
    counts = {"North": 0, "South": 0, "East": 0, "West": 0}
    if result.boxes is None:
        return counts
    vehicle_classes = {2, 3, 5, 7}  # COCO: car, motorcycle, bus, truck
    for box in result.boxes:
        if int(box.cls[0].item()) not in vehicle_classes:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if 0.30 * width <= cx <= 0.70 * width:
            if cy < 0.45 * height:
                counts["North"] += 1
            elif cy > 0.55 * height:
                counts["South"] += 1
        if 0.30 * height <= cy <= 0.70 * height:
            if cx < 0.45 * width:
                counts["West"] += 1
            elif cx > 0.55 * width:
                counts["East"] += 1
    return counts


def adaptive_split(ns_queue, ew_queue, fixed_cycle, min_s, max_s):
    """Allocate a feasible fixed-length adaptive cycle in proportion to smoothed queues."""
    cycle = int(np.clip(fixed_cycle, 2 * min_s, 2 * max_s))
    total = ns_queue + ew_queue
    ns_share = 0.5 if total <= 0 else ns_queue / total
    feasible_max = min(max_s, cycle - min_s)
    ns_s = float(np.clip(cycle * ns_share, min_s, feasible_max))
    ew_s = cycle - ns_s
    # Integer seconds while preserving the cycle length.
    ns_s = int(round(ns_s))
    ew_s = cycle - ns_s
    priority = "Balanced" if abs(ns_queue - ew_queue) < 0.5 else ("N/S" if ns_queue > ew_queue else "E/W")
    return ns_s, ew_s, cycle, priority


if uploaded:
    if st.button("Analyze video", type="primary"):
        try:
            model = load_model()
        except Exception as exc:
            st.error(f"Could not load YOLO. Check the installation and model weights. Details: {exc}")
            st.stop()

        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Could not read this video. Try an MP4 encoded with H.264.")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_s = total_frames / fps if total_frames else 0
        cycle_baseline = 2 * fixed_green
        st.write(f"Clip: {duration_s:.1f}s · sequential local inference · sampling every {frame_step} frames")
        progress = st.progress(0)
        preview = st.empty()
        live_state = st.empty()
        rows = []
        frame_idx = 0
        analyzed = 0
        smooth_ns = None
        smooth_ew = None
        started = time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % frame_step == 0:
                    result = model.predict(frame, conf=confidence, verbose=False)[0]
                    height, width = frame.shape[:2]
                    counts = approach_counts(result, width, height)
                    ns_raw = counts["North"] + counts["South"]
                    ew_raw = counts["East"] + counts["West"]
                    # EMA dampens frame-to-frame detector jitter while preserving changing demand.
                    smooth_ns = float(ns_raw) if smooth_ns is None else 0.65 * smooth_ns + 0.35 * ns_raw
                    smooth_ew = float(ew_raw) if smooth_ew is None else 0.65 * smooth_ew + 0.35 * ew_raw
                    ns_green, ew_green, cycle, priority = adaptive_split(
                        smooth_ns, smooth_ew, cycle_baseline, min_green, max_green
                    )
                    timestamp = frame_idx / fps
                    rows.append(
                        {
                            "time_s": timestamp,
                            **counts,
                            "NS_queue_estimate": ns_raw,
                            "EW_queue_estimate": ew_raw,
                            "NS_smoothed": round(smooth_ns, 2),
                            "EW_smoothed": round(smooth_ew, 2),
                            "NS_green_s": ns_green,
                            "EW_green_s": ew_green,
                            "priority_phase": priority,
                        }
                    )

                    annotated = result.plot()
                    cv2.rectangle(annotated, (int(width * .30), 0), (int(width * .70), int(height * .45)), (0, 220, 255), 2)
                    cv2.rectangle(annotated, (int(width * .30), int(height * .55)), (int(width * .70), height), (0, 220, 255), 2)
                    cv2.rectangle(annotated, (0, int(height * .30)), (int(width * .45), int(height * .70)), (255, 180, 0), 2)
                    cv2.rectangle(annotated, (int(width * .55), int(height * .30)), (width, int(height * .70)), (255, 180, 0), 2)
                    cv2.putText(
                        annotated,
                        f"N/S {ns_raw} | E/W {ew_raw}  ->  green {ns_green}s / {ew_green}s",
                        (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA
                    )
                    cv2.putText(
                        annotated,
                        f"Next priority: {priority}",
                        (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA
                    )
                    cv2.putText(annotated, f"N/S {ns_raw} | E/W {ew_raw}  ->  green {ns_green}s / {ew_green}s", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(annotated, f"Next priority: {priority}", (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Video time {timestamp:.1f}s", width="stretch")
                    live_state.info(
                        f"At {timestamp:.1f}s · estimated queues N/S {ns_raw}, E/W {ew_raw} · "
                        f"recommended split {ns_green}s / {ew_green}s · next priority {priority}"
                    )
                    analyzed += 1
                    if total_frames:
                        progress.progress(min(frame_idx / total_frames, 1.0))
                frame_idx += 1
        finally:
            cap.release()
            try:
                Path(video_path).unlink(missing_ok=True)
            except OSError:
                pass

        progress.progress(1.0)
        if not rows:
            st.error("No frames were analyzed.")
            st.stop()

        df = pd.DataFrame(rows)
        avg_ns = float(df["NS_queue_estimate"].mean())
        avg_ew = float(df["EW_queue_estimate"].mean())
        avg_ns_green = int(round(df["NS_green_s"].mean()))
        avg_ew_green = int(round(df["EW_green_s"].mean()))
        priority_share = float((df["priority_phase"] == "N/S").mean() * 100)
        st.subheader("Results")
        a, b, c, d = st.columns(4)
        a.metric("Avg N/S detections", f"{avg_ns:.1f}")
        b.metric("Avg E/W detections", f"{avg_ew:.1f}")
        c.metric("Typical adaptive split", f"{avg_ns_green}s / {avg_ew_green}s")
        d.metric("Samples prioritizing N/S", f"{priority_share:.0f}%")
        st.caption(
            f"Fixed baseline: {fixed_green}s / {fixed_green}s ({cycle_baseline}s cycle) · "
            f"Adaptive recommendations update per sampled frame · {analyzed} frames analyzed in {time.time() - started:.1f}s"
        )

        queues_tab, controller_tab = st.tabs(["Queue estimates", "Controller response"])
        with queues_tab:
            st.line_chart(df.set_index("time_s")[["NS_queue_estimate", "EW_queue_estimate"]], height=340)
        with controller_tab:
            st.line_chart(df.set_index("time_s")[["NS_green_s", "EW_green_s"]], height=340)
            st.caption("Adaptive green splits are calculated from an exponentially smoothed vehicle count at each sampled video frame.")

        comparison = pd.DataFrame(
            {
                "Policy": ["Fixed baseline", "Adaptive recommendation"],
                "Green allocation": [f"{fixed_green}s / {fixed_green}s", f"about {avg_ns_green}s / {avg_ew_green}s"],
                "Decision basis": ["Equal split", "Smoothed N/S and E/W detections per sampled frame"],
            }
        )
        st.dataframe(comparison, hide_index=True, width="stretch")
        st.warning(
            "Counts are detections in simple image regions, not tracked unique vehicles or validated queue lengths. "
            "This demo recommends timing; it does not operate a signal or measure actual delay reduction."
        )
        st.download_button("Download frame-level results (CSV)", df.to_csv(index=False), "queuesense_counts.csv", "text/csv")
else:
    st.markdown("### Demo flow")
    st.markdown(
        "1. Upload a short, stable, preferably elevated view of a four-way intersection.\n"
        "2. QueueSense runs YOLOv8 nano locally and marks four approach regions.\n"
        "3. Watch the recommended green split update as sampled frames are analyzed.\n"
        "4. Review queue and controller-response charts, then export the frame-level CSV."
    )
    st.caption("For the clearest demo, choose a clip where all four approaches are visible and tune the simple image regions to that camera view.")
