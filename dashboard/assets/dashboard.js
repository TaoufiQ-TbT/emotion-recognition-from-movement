/* Dashboard behavior: uploading a video, and the charts that follow it during playback.
   Dash loads every .js file in assets/ automatically. */
(function () {
  "use strict";

  const MAX_UPLOAD_BYTES = 300 * 1024 * 1024;
  const CHUNK_BYTES = 10 * 1024 * 1024;
  const STEP_KEYS = ["upload", "convert", "poses", "movement", "predict"];

  // Display names of the model's labels, in display order.
  const EMOTION_NAMES = {
    happiness: "Happiness",
    surprised: "Surprise",
    fear: "Fear",
    sadness: "Sadness",
    disgusted: "Disgust",
    angry: "Anger",
  };
  const EMOTION_ORDER = Object.keys(EMOTION_NAMES);

  const $ = (id) => document.getElementById(id);
  const setProps = (id, props) => window.dash_clientside.set_props(id, props);
  const emotionName = (key) => EMOTION_NAMES[key] || key.charAt(0).toUpperCase() + key.slice(1);
  const fillColor = (key) => `var(--emo-${key}, var(--text-3))`;
  const textColor = (key) => `var(--emo-${key}-text, var(--text))`;
  const percent = (value) => `${Math.round(value * 100)}%`;
  const argmax = (probs) => Object.keys(probs).reduce((a, b) => (probs[b] > probs[a] ? b : a));
  const clamp01 = (v) => Math.min(1, Math.max(0, v));

  // A playback time, as m:ss.
  function clock(seconds) {
    const s = Math.max(0, Math.floor(seconds || 0));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }

  // A length of time: "13.6 s" under a minute, "1:05" from a minute.
  function span(seconds) {
    if (seconds < 0.05) return "0 s";
    return seconds < 60 ? `${seconds.toFixed(1)} s` : clock(seconds);
  }

  // Create an element with an optional class and text, and add it to a parent.
  function el(tag, className, parent, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }

  // The small color dot that identifies an emotion.
  function dot(parent, key) {
    el("i", "dot", parent).style.background = fillColor(key);
  }

  // ---------------------------------------------------------------------------
  // Upload
  // ---------------------------------------------------------------------------

  const isBusy = () => ($("app-main") || {}).className === "main state-processing";

  // Show the error view after a failed upload.
  function showUploadError(message) {
    setProps("app-main", { className: "main state-error" });
    setProps("error-message", { children: message });
    setProps("status", { className: "status status-error" });
    setProps("status-text", { children: "Upload failed" });
  }

  // Open the file picker, unless an analysis is already running.
  function chooseFile() {
    if (isBusy()) return;
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "video/*,.ts,.mkv";
    input.addEventListener("change", () => input.files[0] && upload(input.files[0]));
    input.click();
  }

  // Send the file in chunks, then tell the server it is complete. app.py starts the analysis.
  async function upload(file) {
    const looksLikeVideo = file.type.startsWith("video/") || /\.(mp4|mov|m4v|avi|mkv|webm|ts)$/i.test(file.name);
    if (!looksLikeVideo) return showUploadError(`“${file.name}” is not a video file.`);
    if (file.size > MAX_UPLOAD_BYTES) {
      return showUploadError(`“${file.name}” is larger than 300 MB. Shorten or compress it, then try again.`);
    }

    const video = $("video-player");
    if (video) video.pause();
    setProps("app-main", { className: "main state-processing" });
    setProps("progress-file", { children: file.name });
    setProps("status", { className: "status status-busy" });
    setProps("status-text", { children: "Uploading…" });
    STEP_KEYS.forEach((key, i) => {
      setProps(`step-${key}`, { className: i === 0 ? "step active" : "step" });
      setProps(`step-${key}-detail`, { children: "" });
    });

    // Same format the server validates: 8 to 64 letters and digits.
    const uploadId = Date.now().toString(36) + Math.random().toString(36).slice(2, 12);
    const chunks = Math.max(1, Math.ceil(file.size / CHUNK_BYTES));
    try {
      for (let i = 0; i < chunks; i++) {
        setProps("step-upload-detail", { children: percent(i / chunks) });
        const response = await fetch("/upload_chunk", {
          method: "POST",
          headers: { "X-Upload-ID": uploadId, "X-Chunk-Index": String(i), "Content-Type": "application/octet-stream" },
          body: file.slice(i * CHUNK_BYTES, (i + 1) * CHUNK_BYTES),
        });
        if (!response.ok) throw new Error(`the server refused part ${i + 1} (${response.status})`);
      }
      setProps("step-upload-detail", { children: "100%" });
      const response = await fetch("/upload_fusion", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ upload_id: uploadId, filename: file.name }),
      });
      if (!response.ok) throw new Error(`the server did not confirm it (${response.status})`);
      const result = await response.json();
      setProps("upload-trigger", { data: { upload_id: result.upload_id, filename: file.name, sent: Date.now() } });
    } catch (error) {
      showUploadError(`The upload failed: ${error.message}. Is the dashboard still running?`);
    }
  }

  document.addEventListener("click", (event) => {
    if (event.target.closest("#choose-video, #new-video, #retry-video, #dropzone")) chooseFile();
    const toggle = event.target.closest("#skeleton-toggle");
    if (toggle) toggleSkeleton(toggle);
    if (event.target.closest("#theme-toggle")) toggleTheme();
  });

  // Drop a file anywhere on the window.
  let dragDepth = 0;
  const carriesFiles = (event) => Array.from((event.dataTransfer && event.dataTransfer.types) || []).includes("Files");
  window.addEventListener("dragenter", (event) => {
    if (!carriesFiles(event) || isBusy()) return;
    event.preventDefault();
    dragDepth += 1;
    document.body.classList.add("is-dragging");
  });
  window.addEventListener("dragover", (event) => {
    if (carriesFiles(event)) event.preventDefault();
  });
  window.addEventListener("dragleave", (event) => {
    if (!carriesFiles(event)) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) document.body.classList.remove("is-dragging");
  });
  window.addEventListener("drop", (event) => {
    if (!carriesFiles(event)) return;
    event.preventDefault();
    dragDepth = 0;
    document.body.classList.remove("is-dragging");
    const file = event.dataTransfer.files[0];
    if (file && !isBusy()) upload(file);
  });

  // ---------------------------------------------------------------------------
  // SVG helpers
  // ---------------------------------------------------------------------------

  const SVG = "http://www.w3.org/2000/svg";

  function svg(tag, attrs, parent) {
    const node = document.createElementNS(SVG, tag);
    for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
    if (parent) parent.appendChild(node);
    return node;
  }

  // ---------------------------------------------------------------------------
  // Curves: a monotone cubic through each window center (Fritsch–Carlson). Monotone means a
  // curve never overshoots between two windows, so it stays within 0–100 %.
  // ---------------------------------------------------------------------------

  function tangents(xs, ys) {
    const n = xs.length;
    if (n < 2) return [0];
    const slopes = [];
    for (let i = 0; i < n - 1; i++) slopes.push((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]));
    const m = new Array(n);
    m[0] = slopes[0];
    m[n - 1] = slopes[n - 2];
    for (let i = 1; i < n - 1; i++) m[i] = slopes[i - 1] * slopes[i] <= 0 ? 0 : (slopes[i - 1] + slopes[i]) / 2;
    for (let i = 0; i < n - 1; i++) {
      if (slopes[i] === 0) {
        m[i] = 0;
        m[i + 1] = 0;
        continue;
      }
      const a = m[i] / slopes[i];
      const b = m[i + 1] / slopes[i];
      const s = a * a + b * b;
      if (s > 9) {
        const t = 3 / Math.sqrt(s);
        m[i] = t * a * slopes[i];
        m[i + 1] = t * b * slopes[i];
      }
    }
    return m;
  }

  // Window centers, plus the start and the end of the video so curves span the whole timeline.
  function buildCurves() {
    const windows = state.data.windows;
    const total = duration();
    const xs = [];
    const rows = [];
    if (windows[0].center > 0) {
      xs.push(0);
      rows.push(windows[0].probs);
    }
    for (const w of windows) {
      if (xs.length && w.center <= xs[xs.length - 1]) continue;
      xs.push(w.center);
      rows.push(w.probs);
    }
    if (total > xs[xs.length - 1]) {
      xs.push(total);
      rows.push(windows[windows.length - 1].probs);
    }
    const curves = { xs, ys: {}, ms: {} };
    for (const key of state.labels) {
      curves.ys[key] = rows.map((p) => p[key] || 0);
      curves.ms[key] = tangents(xs, curves.ys[key]);
    }
    state.curves = curves;
  }

  // Value of one emotion's curve at a time: a cubic Hermite segment between two window centers.
  function curveAt(key, time) {
    const { xs } = state.curves;
    const ys = state.curves.ys[key];
    const ms = state.curves.ms[key];
    if (xs.length < 2 || time <= xs[0]) return ys[0];
    if (time >= xs[xs.length - 1]) return ys[ys.length - 1];
    let i = 0;
    while (xs[i + 1] < time) i++;
    const h = xs[i + 1] - xs[i];
    const t = (time - xs[i]) / h;
    const t2 = t * t;
    const t3 = t2 * t;
    return (
      (2 * t3 - 3 * t2 + 1) * ys[i] +
      (t3 - 2 * t2 + t) * h * ms[i] +
      (-2 * t3 + 3 * t2) * ys[i + 1] +
      (t3 - t2) * h * ms[i + 1]
    );
  }

  // ---------------------------------------------------------------------------
  // Results that follow the video, redrawn every frame
  // ---------------------------------------------------------------------------

  const state = { data: null, labels: [], curves: null, shown: null, frame: 0, rings: null, chart: null, resize: null };

  // Entry point for a finished analysis: build everything for this result.
  function load(data) {
    if (!data || !data.windows || !data.windows.length) return;
    const keys = Object.keys(data.windows[0].probs);
    state.labels = EMOTION_ORDER.filter((k) => keys.includes(k)).concat(keys.filter((k) => !EMOTION_ORDER.includes(k)));
    state.data = {
      ...data,
      windows: data.windows.map((w) => ({ ...w, center: (w.start + w.end) / 2, top: argmax(w.probs) })),
    };
    state.shown = null;
    skeleton.url = data.poses || null;
    skeleton.frames = null;
    if (skeleton.on) loadPoses();

    buildRings();
    buildTable();
    refreshForDuration();
    attachVideo();
    redraw();
  }

  // Everything that depends on the video length, which is exact once the video's metadata loads.
  function refreshForDuration() {
    buildCurves();
    drawChart();
    renderSummary();
  }

  function duration() {
    const video = $("video-player");
    if (video && Number.isFinite(video.duration) && video.duration > 0) return video.duration;
    const windows = state.data.windows;
    return windows[windows.length - 1].end;
  }

  function blend(a, b, t) {
    const out = {};
    for (const key of state.labels) out[key] = (a[key] || 0) + ((b[key] || 0) - (a[key] || 0)) * t;
    return out;
  }

  // Readings at a playback time. "Now" follows the curves (renormalized so it adds up to 100 %);
  // "so far" blends the running averages of the two windows around that time.
  function valuesAt(time) {
    const raw = {};
    let sum = 0;
    for (const key of state.labels) {
      raw[key] = clamp01(curveAt(key, time));
      sum += raw[key];
    }
    const probs = {};
    for (const key of state.labels) probs[key] = sum > 0 ? raw[key] / sum : 0;

    const windows = state.data.windows;
    let cumulative = windows[windows.length - 1].cumulative;
    if (time <= windows[0].center) cumulative = windows[0].cumulative;
    else {
      for (let i = 1; i < windows.length; i++) {
        if (time <= windows[i].center) {
          const a = windows[i - 1];
          const b = windows[i];
          cumulative = blend(a.cumulative, b.cumulative, (time - a.center) / (b.center - a.center || 1));
          break;
        }
      }
    }
    return { raw, probs, cumulative };
  }

  // Move the displayed values a share of the way to the target each frame: playback stays fluid
  // and jumps (seeking) glide. Returns whether anything is still moving.
  function approach(shown, target, share) {
    let moving = false;
    for (const key of state.labels) {
      const gap = target[key] - shown[key];
      if (Math.abs(gap) > 0.0005) {
        shown[key] += gap * share;
        moving = true;
      } else {
        shown[key] = target[key];
      }
    }
    return moving;
  }

  // One animation frame. It keeps running while the video plays or the values are still settling.
  function tick() {
    state.frame = 0;
    if (!state.data) return;
    const video = $("video-player");
    const time = video ? video.currentTime || 0 : 0;
    const target = valuesAt(time);
    if (!state.shown) state.shown = { probs: { ...target.probs }, cumulative: { ...target.cumulative } };
    const share = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 1 : 0.2;
    const nowMoving = approach(state.shown.probs, target.probs, share);
    const mixMoving = approach(state.shown.cumulative, target.cumulative, share);
    render(time, target.raw);
    if (nowMoving || mixMoving || (video && !video.paused)) state.frame = requestAnimationFrame(tick);
  }

  // Ask for one more frame, unless one is already scheduled.
  function redraw() {
    if (!state.frame) state.frame = requestAnimationFrame(tick);
  }

  // ---------------------------------------------------------------------------
  // Rings: outer = average since the start, inner (thinner) = now
  // ---------------------------------------------------------------------------

  // A wide outer ring and a thin inner one, with clear air between them.
  const OUTER = { radius: 44.5, width: 11, gap: 1.2 };
  const INNER = { radius: 32.5, width: 4, gap: 1.8 };

  // Create the two rings for this result. setRing then updates their arcs on every frame.
  function buildRings() {
    const box = $("emotion-rings");
    box.textContent = "";
    const root = svg("svg", { viewBox: "0 0 100 100", "aria-hidden": "true" }, box);
    const ring = ({ radius, width }) => {
      svg("circle", { cx: 50, cy: 50, r: radius, class: "ring-track", "stroke-width": width }, root);
      return state.labels.map(() => {
        const arc = svg("circle", { cx: 50, cy: 50, r: radius, class: "ring-arc", "stroke-width": width, pathLength: 100 }, root);
        arc.style.strokeDasharray = "0 100";
        return arc;
      });
    };
    state.rings = { outer: ring(OUTER), inner: ring(INNER) };
  }

  // One arc per emotion, laid end to end, with a small gap between arcs. Shares under 1 % are
  // not drawn: the table shows them as 0 %, and a sliver would look like a rendering glitch.
  function setRing(arcs, values, gap) {
    let start = 0;
    arcs.forEach((arc, i) => {
      const key = state.labels[i];
      const share = (values[key] || 0) * 100;
      const length = share >= 1 ? Math.max(0, share - gap) : 0;
      arc.style.stroke = fillColor(key);
      arc.style.strokeDasharray = `${length} ${100 - length}`;
      arc.style.strokeDashoffset = String(-(start + gap / 2));
      start += share;
    });
  }

  // One row per emotion, with its Now and So far cells.
  function buildTable() {
    const body = $("emotion-table-body");
    body.textContent = "";
    for (const key of state.labels) {
      const row = el("tr", "", body);
      row.dataset.emotion = key;
      const name = el("span", "emotion-name", el("td", "", row));
      dot(name, key);
      name.appendChild(document.createTextNode(emotionName(key)));
      el("td", "cell-now", row, "0%");
      el("td", "cell-mix", row, "0%");
    }
  }

  // ---------------------------------------------------------------------------
  // Timeline chart, redrawn whenever its size changes
  // ---------------------------------------------------------------------------

  const MARGIN = { left: 40, right: 8, top: 8, bottom: 22 };

  // Draw the axes and curves. render() only moves the playhead and the dots.
  function drawChart() {
    const box = $("timeline");
    const width = box.clientWidth;
    const height = box.clientHeight;
    if (!state.curves || width < 50 || height < 50) return;
    box.textContent = "";

    const total = duration();
    const plotW = width - MARGIN.left - MARGIN.right;
    const plotH = height - MARGIN.top - MARGIN.bottom;
    const X = (t) => MARGIN.left + (t / total) * plotW;
    const Y = (v) => MARGIN.top + (1 - v) * plotH;
    const root = svg("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img" }, box);
    root.setAttribute("aria-label", "Probability of each emotion over the video, from 0 to 100 percent");

    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      svg("line", { x1: MARGIN.left, x2: width - MARGIN.right, y1: Y(v), y2: Y(v), class: "chart-grid" }, root);
      if (v === 0 || v === 0.5 || v === 1) {
        const label = svg("text", { x: MARGIN.left - 8, y: Y(v), "text-anchor": "end", "dominant-baseline": "middle", class: "chart-label" }, root);
        label.textContent = percent(v);
      }
    }
    for (let i = 0; i <= 4; i++) {
      const t = (total * i) / 4;
      const anchor = i === 0 ? "start" : i === 4 ? "end" : "middle";
      const label = svg("text", { x: X(t), y: height - 4, "text-anchor": anchor, class: "chart-label" }, root);
      label.textContent = clock(t);
    }

    const { xs } = state.curves;
    const lines = {};
    const dots = {};
    for (const key of state.labels) {
      const ys = state.curves.ys[key];
      const ms = state.curves.ms[key];
      // An emotion that never reaches 2 % would only redraw the 0 % line; the table still lists it.
      if (Math.max(...ys) < 0.02) continue;
      let d = `M ${X(xs[0])} ${Y(ys[0])}`;
      for (let i = 0; i < xs.length - 1; i++) {
        const h = xs[i + 1] - xs[i];
        d +=
          ` C ${X(xs[i] + h / 3)} ${Y(ys[i] + (ms[i] * h) / 3)}` +
          ` ${X(xs[i + 1] - h / 3)} ${Y(ys[i + 1] - (ms[i + 1] * h) / 3)}` +
          ` ${X(xs[i + 1])} ${Y(ys[i + 1])}`;
      }
      const area = svg("path", { d: `${d} L ${X(xs[xs.length - 1])} ${Y(0)} L ${X(xs[0])} ${Y(0)} Z`, class: "chart-area" }, root);
      area.style.fill = fillColor(key);
      lines[key] = svg("path", { d, class: "chart-line" }, root);
      lines[key].style.stroke = fillColor(key);
    }
    const playhead = svg("line", { y1: MARGIN.top - 4, y2: MARGIN.top + plotH, class: "chart-playhead" }, root);
    for (const key of state.labels) {
      dots[key] = svg("circle", { r: 4.5, class: "chart-dot" }, root);
      dots[key].style.fill = fillColor(key);
    }

    // Click or drag anywhere on the plot to move through the video.
    const hit = svg("rect", { x: MARGIN.left, y: 0, width: plotW, height, fill: "transparent" }, root);
    const seek = (event) => {
      const video = $("video-player");
      const bounds = root.getBoundingClientRect();
      const ratio = clamp01((event.clientX - bounds.left - MARGIN.left) / plotW);
      if (video) video.currentTime = ratio * total;
      redraw();
    };
    hit.addEventListener("pointerdown", (event) => {
      hit.setPointerCapture(event.pointerId);
      seek(event);
    });
    hit.addEventListener("pointermove", (event) => {
      if (hit.hasPointerCapture(event.pointerId)) seek(event);
    });

    state.chart = { X, Y, lines, dots, playhead };
    if (state.shown) redraw();
  }

  // Share of the video where each emotion was the strongest, and the sidebar summary.
  function renderSummary() {
    const windows = state.data.windows;
    const total = duration();
    const seconds = {};
    windows.forEach((w, i) => {
      // Each window owns the time closer to its center than to its neighbors'.
      const from = i === 0 ? 0 : (windows[i - 1].center + w.center) / 2;
      const to = i === windows.length - 1 ? total : (w.center + windows[i + 1].center) / 2;
      if (to > from) seconds[w.top] = (seconds[w.top] || 0) + (to - from);
    });
    const shares = {};
    for (const key of state.labels) shares[key] = (seconds[key] || 0) / total;

    const box = $("share-bars");
    box.textContent = "";
    const top = argmax(shares);
    for (const key of state.labels.slice().sort((a, b) => shares[b] - shares[a])) {
      const row = el("div", key === top ? "bar-row is-top" : "bar-row", box);
      const name = el("span", "bar-name", row);
      dot(name, key);
      name.appendChild(document.createTextNode(emotionName(key)));
      const fill = el("span", "bar-fill", el("span", "bar-track", row));
      fill.style.background = fillColor(key);
      fill.style.width = `${shares[key] * 100}%`;
      el("span", "bar-value", row, span(seconds[key] || 0));
    }

    // The model's verdict, computed on the server like the published evaluation. The rings show the mix.
    $("overall-final").textContent = `Whole video: ${emotionName(state.data.wholeVideo)}`;
    const notices = state.data.notices || [];
    const notice = $("quality-notice");
    notice.textContent = notices.length ? `Less reliable: ${notices.join(" ")}` : "";
    notice.hidden = !notices.length;
    $("video-name").textContent = state.data.filename || "—";
    $("video-name").title = state.data.filename || "";
    $("video-duration").textContent = clock(total);
    $("video-windows").textContent = String(windows.length);
  }

  // Show an emotion name in its color. The check avoids touching the page on every frame.
  function setEmotion(id, key) {
    const node = $(id);
    if (node.dataset.emotion === key) return;
    node.dataset.emotion = key;
    node.textContent = emotionName(key);
    node.style.color = textColor(key);
  }

  // Show the values of the current frame: rings, table, chart markers and skeleton.
  function render(time, raw) {
    const { probs, cumulative } = state.shown;
    const top = argmax(probs);
    const mixTop = argmax(cumulative);

    $("now-time").textContent = clock(time);
    setEmotion("now-emotion", top);
    $("now-value").textContent = percent(probs[top]);
    setRing(state.rings.outer, cumulative, OUTER.gap);
    setRing(state.rings.inner, probs, INNER.gap);
    $("emotion-rings").setAttribute("role", "img");
    $("emotion-rings").setAttribute(
      "aria-label",
      `Now: ${state.labels.map((k) => `${emotionName(k)} ${percent(probs[k])}`).join(", ")}. ` +
        `Since the start: ${state.labels.map((k) => `${emotionName(k)} ${percent(cumulative[k])}`).join(", ")}.`
    );

    for (const row of $("emotion-table-body").children) {
      const key = row.dataset.emotion;
      const now = row.children[1];
      const mix = row.children[2];
      now.textContent = percent(probs[key]);
      mix.textContent = percent(cumulative[key]);
      now.classList.toggle("is-top", key === top);
      mix.classList.toggle("is-top", key === mixTop);
    }

    if (state.chart) {
      const { X, Y, lines, dots, playhead } = state.chart;
      const x = X(time);
      playhead.setAttribute("x1", x);
      playhead.setAttribute("x2", x);
      for (const key of state.labels) {
        // Dots sit on the curves; values under 2 % stay hidden to keep the baseline clean.
        dots[key].setAttribute("cx", x);
        dots[key].setAttribute("cy", Y(raw[key]));
        dots[key].style.display = raw[key] >= 0.02 ? "" : "none";
        if (lines[key]) lines[key].classList.toggle("is-top", key === top);
      }
    }

    drawSkeleton(time, top);
  }

  // Connect to the video player once: the same element is reused between analyses.
  function attachVideo() {
    const video = $("video-player");
    if (!video || video.dataset.synced) return;
    video.dataset.synced = "1";
    ["play", "pause", "seeked", "timeupdate", "ended"].forEach((name) => video.addEventListener(name, redraw));
    video.addEventListener("loadedmetadata", () => {
      if (state.data) refreshForDuration();
      redraw();
    });
    state.resize = new ResizeObserver(() => state.data && drawChart());
    state.resize.observe($("timeline"));
    new ResizeObserver(redraw).observe(video.parentElement);
  }

  // ---------------------------------------------------------------------------
  // Skeleton over the video, from the 2D landmarks saved by the analysis
  // ---------------------------------------------------------------------------

  // MediaPipe Pose body connections. Face points are left out; the nose is drawn as the head.
  const BONES = [
    [11, 12], [11, 13], [13, 15], [15, 19], [12, 14], [14, 16], [16, 20],
    [11, 23], [12, 24], [23, 24],
    [23, 25], [25, 27], [27, 29], [29, 31], [27, 31],
    [24, 26], [26, 28], [28, 30], [30, 32], [28, 32],
  ];
  const JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
  const skeleton = { on: false, url: null, frames: null, loading: false, colors: {} };

  // The poses are downloaded the first time the overlay is turned on.
  function toggleSkeleton(button) {
    skeleton.on = !skeleton.on;
    button.classList.toggle("is-on", skeleton.on);
    button.setAttribute("aria-pressed", String(skeleton.on));
    button.title = skeleton.on ? "Hide the detected skeleton" : "Show the detected skeleton";
    if (skeleton.on) loadPoses();
    redraw();
  }

  async function loadPoses() {
    if (skeleton.frames || skeleton.loading || !skeleton.url) return;
    skeleton.loading = true;
    try {
      const response = await fetch(skeleton.url);
      if (response.ok) skeleton.frames = (await response.json()).frames;
    } finally {
      skeleton.loading = false;
      redraw();
    }
  }

  // Landmarks at a playback time, blended between the two nearest saved poses (one every few
  // video frames). Null when no pose is close: the person was not detected there.
  function poseAt(time) {
    const frames = skeleton.frames;
    if (!frames || !frames.length) return null;
    let lo = 0;
    let hi = frames.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (frames[mid][0] < time) lo = mid + 1;
      else hi = mid;
    }
    if (lo === 0) return frames[0][0] - time < 0.5 ? frames[0][1] : null;
    if (lo === frames.length) return time - frames[lo - 1][0] < 0.5 ? frames[lo - 1][1] : null;
    const [ta, a] = frames[lo - 1];
    const [tb, b] = frames[lo];
    if (tb - ta > 1) {
      const [tn, near] = time - ta < tb - time ? frames[lo - 1] : frames[lo];
      return Math.abs(tn - time) < 0.3 ? near : null;
    }
    const t = (time - ta) / (tb - ta);
    return a.map((v, i) => v + (b[i] - v) * t);
  }

  // Draw the pose over the video, in the color of the current top emotion.
  function drawSkeleton(time, key) {
    const canvas = $("skeleton-canvas");
    const video = $("video-player");
    if (!canvas || !video) return;
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const pose = skeleton.on ? poseAt(time) : null;
    if (!pose || !video.videoWidth) return;

    // Where the picture sits inside the player (object-fit: contain).
    const scale = Math.min(width / video.videoWidth, height / video.videoHeight);
    const w = video.videoWidth * scale;
    const h = video.videoHeight * scale;
    const left = (width - w) / 2;
    const top = (height - h) / 2;
    const point = (i) => [left + pose[i * 3] * w, top + pose[i * 3 + 1] * h];
    const visible = (i) => pose[i * 3 + 2] >= 0.4;
    if (!skeleton.colors[key]) {
      skeleton.colors[key] = getComputedStyle(document.documentElement).getPropertyValue(`--emo-${key}`).trim() || "#ffffff";
    }
    const color = skeleton.colors[key];
    const size = Math.max(2, Math.min(5, w / 220));

    // Bones: a soft dark outline first, so the colored line reads on any background.
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (const [stroke, lineWidth] of [["rgba(0, 0, 0, 0.45)", size + 3], [color, size]]) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      for (const [a, b] of BONES) {
        if (!visible(a) || !visible(b)) continue;
        ctx.moveTo(...point(a));
        ctx.lineTo(...point(b));
      }
      ctx.stroke();
    }
    for (const i of JOINTS) {
      if (!visible(i)) continue;
      const [x, y] = point(i);
      ctx.beginPath();
      ctx.arc(x, y, i === 0 ? size * 2 : size * 0.95, 0, 2 * Math.PI);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = color;
      ctx.stroke();
    }
  }

  // ---------------------------------------------------------------------------
  // Light and dark appearance (the first theme is set by INDEX_STRING in app.py)
  // ---------------------------------------------------------------------------

  function toggleTheme() {
    const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("theme", theme);
    } catch {
      // Storage is blocked: the choice lasts until the page is reloaded.
    }
    // Canvas needs real colors; the CSS variables change with the theme.
    skeleton.colors = {};
    redraw();
  }

  // Called by the clientside callback in app.py whenever a new analysis is ready.
  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    emotion: {
      load: function (data) {
        load(data);
        return window.dash_clientside.no_update;
      },
    },
  });
})();
