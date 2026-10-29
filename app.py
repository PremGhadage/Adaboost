import os
import pickle
import warnings

import numpy as np
from flask import Flask, jsonify, render_template_string, request

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the model once (reused across requests on the same serverless instance)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "decision.pkl"), "rb") as f:
    model = pickle.load(f)

# ---------------------------------------------------------------------------
# Edit these texts to match what your model predicts
# ---------------------------------------------------------------------------
APP_TITLE = "Admission Predictor"
APP_SUBTITLE = "Enter the student profile and get an instant, easy-to-read prediction."
POSITIVE_TEXT = "likely to be selected"
NEGATIVE_TEXT = "less likely to be selected"

# ---------------------------------------------------------------------------
# Form definition (single source of truth for UI + validation)
# ---------------------------------------------------------------------------
def num(key, label, lo, hi, step, default, hint="", kind="number"):
    return dict(key=key, label=label, min=lo, max=hi, step=step,
                default=default, hint=hint, kind=kind)

GROUPS = [
    ("Personal details", "👤", [
        num("age", "Age", 14, 35, 1, 18),
        dict(key="gender", label="Gender", kind="select", default=0, hint="",
             options=[[0, "Male"], [1, "Female"], [2, "Other"]]),
        num("state", "State code", 0, 99, 1, 0, "Numeric code used while training"),
        num("family_income", "Family income (per year)", 0, 10000000, 1000, 60000),
    ]),
    ("Academics", "🎓", [
        num("high_school_gpa", "High school GPA", 0, 4, 0.01, 3.4, "Out of 4.0", "range"),
        num("sat_score", "SAT score", 400, 1600, 10, 1150, "400 - 1600", "range"),
        num("act_score", "ACT score", 1, 36, 1, 24, "1 - 36", "range"),
        num("attendance_rate", "Attendance rate (%)", 0, 100, 1, 92, "", "range"),
        num("ap_courses", "AP courses taken", 0, 15, 1, 3),
    ]),
    ("Activities & skills", "🏆", [
        num("extracurricular_count", "Extracurricular activities", 0, 15, 1, 3),
        num("volunteer_hours", "Volunteer hours", 0, 500, 1, 60),
        num("leadership_positions", "Leadership positions", 0, 10, 1, 1),
        num("coding_projects", "Coding projects", 0, 20, 1, 1),
        num("online_certifications", "Online certifications", 0, 20, 1, 1),
        num("social_media_hours", "Social media hours / day", 0, 16, 0.5, 3),
    ]),
    ("Evaluations", "📝", [
        num("essay_score", "Essay score", 0, 100, 1, 70, "Out of 100", "range"),
        num("recommendation_score", "Recommendation score", 0, 100, 1, 70, "Out of 100", "range"),
        num("interview_score", "Interview score", 0, 100, 1, 70, "Out of 100", "range"),
    ]),
]

FIELDS = [f for _, _, fields in GROUPS for f in fields]
FIELD_MAP = {f["key"]: f for f in FIELDS}

# Feature order must match training order
ORDER = list(getattr(model, "feature_names_in_", [f["key"] for f in FIELDS]))

# Benchmarks used only to give friendly feedback (strengths / areas to improve)
BENCHMARKS = {
    "high_school_gpa": ("High school GPA", 3.4),
    "sat_score": ("SAT score", 1100),
    "interview_score": ("Interview score", 75),
    "essay_score": ("Essay score", 75),
    "recommendation_score": ("Recommendation score", 75),
    "extracurricular_count": ("Extracurricular activities", 4),
    "ap_courses": ("AP courses", 3),
    "leadership_positions": ("Leadership positions", 1),
    "volunteer_hours": ("Volunteer hours", 90),
    "coding_projects": ("Coding projects", 3),
    "online_certifications": ("Online certifications", 3),
}

PRESETS = {
    "Strong profile": dict(age=18, gender=0, state=10, family_income=80000, high_school_gpa=3.9,
        sat_score=1500, act_score=34, attendance_rate=98, ap_courses=8, extracurricular_count=6,
        volunteer_hours=150, leadership_positions=2, coding_projects=4, social_media_hours=2,
        online_certifications=4, essay_score=90, recommendation_score=90, interview_score=90),
    "Average profile": dict(age=18, gender=0, state=10, family_income=60000, high_school_gpa=3.3,
        sat_score=1150, act_score=24, attendance_rate=92, ap_courses=3, extracurricular_count=3,
        volunteer_hours=60, leadership_positions=1, coding_projects=1, social_media_hours=4,
        online_certifications=1, essay_score=70, recommendation_score=70, interview_score=70),
    "Weak profile": dict(age=18, gender=0, state=10, family_income=40000, high_school_gpa=2.4,
        sat_score=900, act_score=17, attendance_rate=80, ap_courses=0, extracurricular_count=1,
        volunteer_hours=10, leadership_positions=0, coding_projects=0, social_media_hours=7,
        online_certifications=0, essay_score=50, recommendation_score=50, interview_score=50),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_input(data):
    values = {}
    for f in FIELDS:
        raw = data.get(f["key"])
        try:
            v = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"Please enter a valid number for “{f['label']}”.")
        if f["kind"] == "select":
            if v not in [o[0] for o in f["options"]]:
                raise ValueError(f"Please choose a valid option for “{f['label']}”.")
        elif not (f["min"] <= v <= f["max"]):
            raise ValueError(f"“{f['label']}” must be between {f['min']} and {f['max']}.")
        values[f["key"]] = v
    return values


def describe(prob):
    if prob >= 0.80:
        return "Very likely", "great"
    if prob >= 0.60:
        return "Likely", "good"
    if prob >= 0.50:
        return "Slightly favourable", "ok"
    if prob >= 0.40:
        return "Borderline", "warn"
    if prob >= 0.20:
        return "Unlikely", "bad"
    return "Very unlikely", "bad"


def build_feedback(values):
    importances = dict(zip(ORDER, getattr(model, "feature_importances_", np.zeros(len(ORDER)))))
    ranked = sorted(BENCHMARKS.items(), key=lambda kv: importances.get(kv[0], 0), reverse=True)
    strengths, improve = [], []
    for key, (label, bench) in ranked:
        v = values[key]
        (strengths if v >= bench else improve).append(f"{label} ({v:g})")
    return strengths[:4], improve[:4]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template_string(
        PAGE, title=APP_TITLE, subtitle=APP_SUBTITLE,
        groups=GROUPS, presets=PRESETS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        values = parse_input(data)
        X = np.array([[values[k] for k in ORDER]], dtype=float)

        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        classes = list(model.classes_)
        prob = float(proba[classes.index(1)]) if 1 in classes else float(max(proba))

        band, tone = describe(prob)
        strengths, improve = build_feedback(values)
        outcome = POSITIVE_TEXT if pred == 1 else NEGATIVE_TEXT

        message = (
            f"Based on the details you entered, this profile is {outcome}. "
            f"The model puts the chance of a positive outcome at {prob * 100:.0f}%, "
            f"which we describe as “{band.lower()}”."
        )
        if band == "Borderline":
            message += " It is close to the decision line, so small improvements could change the result."

        return jsonify(ok=True, prediction=pred, probability=round(prob, 4),
                       percent=round(prob * 100, 1), band=band, tone=tone,
                       message=message, strengths=strengths, improve=improve)
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 400
    except Exception:
        return jsonify(ok=False, error="Something went wrong while predicting. Please try again."), 500


@app.route("/health")
def health():
    return jsonify(status="ok", features=ORDER)


# ---------------------------------------------------------------------------
# UI (single-file template)
# ---------------------------------------------------------------------------
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<script>
  (function () {
    var t = null;
    try { t = localStorage.getItem("theme"); } catch (e) {}
    if (!t) t = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", t);
  })();
</script>
<style>
  :root {
    --bg: #f4f6fb; --card: #ffffff; --text: #0f172a; --muted: #64748b; --border: #e2e8f0;
    --primary: #4f46e5; --primary-2: #7c3aed; --input: #f8fafc;
    --great: #16a34a; --good: #22c55e; --ok: #eab308; --warn: #f59e0b; --bad: #ef4444;
    --shadow: 0 10px 30px rgba(15, 23, 42, .08);
  }
  :root[data-theme="dark"] {
    --bg: #0b1020; --card: #131a2e; --text: #e5e9f5; --muted: #94a3b8; --border: #243052;
    --primary: #818cf8; --primary-2: #a78bfa; --input: #0f1629;
    --shadow: 0 10px 30px rgba(0, 0, 0, .45);
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0; font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text); transition: background .3s, color .3s; line-height: 1.5;
  }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 24px 18px 60px; }
  header { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 22px; }
  .brand { display: flex; align-items: center; gap: 12px; }
  .logo { width: 44px; height: 44px; border-radius: 12px; display: grid; place-items: center; font-size: 22px;
    background: linear-gradient(135deg, var(--primary), var(--primary-2)); color: #fff; }
  h1 { font-size: 1.35rem; margin: 0; }
  .sub { color: var(--muted); font-size: .92rem; margin: 2px 0 0; }
  .icon-btn { border: 1px solid var(--border); background: var(--card); color: var(--text); border-radius: 12px;
    padding: 10px 14px; cursor: pointer; font-size: .95rem; box-shadow: var(--shadow); transition: transform .15s; }
  .icon-btn:hover { transform: translateY(-1px); }
  .grid { display: grid; grid-template-columns: 1.4fr 1fr; gap: 20px; align-items: start; }
  @media (max-width: 920px) { .grid { grid-template-columns: 1fr; } }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 18px; padding: 20px;
    box-shadow: var(--shadow); transition: background .3s, border-color .3s; }
  .card + .card { margin-top: 16px; }
  .card h2 { font-size: 1.02rem; margin: 0 0 14px; display: flex; align-items: center; gap: 8px; }
  .fields { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px 16px; }
  @media (max-width: 600px) { .fields { grid-template-columns: 1fr; } }
  label { display: block; font-size: .84rem; font-weight: 600; margin-bottom: 6px; }
  .hint { font-weight: 400; color: var(--muted); font-size: .76rem; }
  input[type=number], select {
    width: 100%; padding: 11px 12px; border-radius: 10px; border: 1px solid var(--border);
    background: var(--input); color: var(--text); font-size: .95rem; outline: none; transition: border .2s, box-shadow .2s;
  }
  input:focus, select:focus { border-color: var(--primary); box-shadow: 0 0 0 3px color-mix(in srgb, var(--primary) 25%, transparent); }
  .range-row { display: flex; align-items: center; gap: 12px; }
  input[type=range] { flex: 1; accent-color: var(--primary); height: 6px; cursor: pointer; }
  .val { min-width: 52px; text-align: center; font-weight: 700; font-size: .85rem; padding: 5px 8px;
    border-radius: 8px; background: var(--input); border: 1px solid var(--border); }
  .presets { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
  .chip { border: 1px solid var(--border); background: var(--input); color: var(--text); padding: 7px 12px;
    border-radius: 999px; cursor: pointer; font-size: .84rem; transition: all .15s; }
  .chip:hover { border-color: var(--primary); color: var(--primary); }
  .actions { display: flex; gap: 10px; margin-top: 18px; flex-wrap: wrap; }
  .btn { border: 0; border-radius: 12px; padding: 13px 20px; font-weight: 700; font-size: .98rem; cursor: pointer; transition: transform .15s, opacity .2s; }
  .btn:hover { transform: translateY(-1px); }
  .btn.primary { flex: 1; color: #fff; background: linear-gradient(135deg, var(--primary), var(--primary-2)); min-width: 180px; }
  .btn.ghost { background: var(--input); color: var(--text); border: 1px solid var(--border); }
  .btn[disabled] { opacity: .6; cursor: wait; }
  .sticky { position: sticky; top: 16px; }
  .empty { text-align: center; color: var(--muted); padding: 30px 10px; }
  .empty .big { font-size: 2.4rem; }
  .gauge { position: relative; width: 170px; height: 170px; margin: 4px auto 10px; }
  .gauge svg { transform: rotate(-90deg); }
  .gauge .track { stroke: var(--border); }
  .gauge .bar { transition: stroke-dashoffset 1s cubic-bezier(.2, .8, .2, 1), stroke .4s; }
  .gauge .center { position: absolute; inset: 0; display: grid; place-items: center; text-align: center; }
  .gauge .pct { font-size: 2.1rem; font-weight: 800; line-height: 1; }
  .gauge .lbl { font-size: .72rem; color: var(--muted); }
  .badge { display: inline-block; padding: 6px 14px; border-radius: 999px; font-weight: 700; font-size: .9rem; color: #fff; }
  .tone-great { background: var(--great); } .tone-good { background: var(--good); }
  .tone-ok { background: var(--ok); color: #1f2937; } .tone-warn { background: var(--warn); color: #1f2937; }
  .tone-bad { background: var(--bad); }
  .center-t { text-align: center; }
  .msg { margin: 14px 0 4px; font-size: .96rem; }
  .list-title { font-size: .82rem; font-weight: 700; margin: 14px 0 6px; }
  .tags { display: flex; flex-wrap: wrap; gap: 6px; }
  .tag { padding: 4px 10px; border-radius: 8px; font-size: .78rem; border: 1px solid var(--border); background: var(--input); }
  .tag.good { border-color: var(--good); } .tag.imp { border-color: var(--warn); }
  .hist { list-style: none; margin: 0; padding: 0; }
  .hist li { display: flex; justify-content: space-between; gap: 10px; padding: 9px 0; border-bottom: 1px dashed var(--border); font-size: .88rem; }
  .hist li:last-child { border-bottom: 0; }
  .error { background: color-mix(in srgb, var(--bad) 14%, transparent); color: var(--bad); border: 1px solid var(--bad);
    padding: 10px 12px; border-radius: 10px; font-size: .88rem; margin-top: 14px; display: none; }
  .toast { position: fixed; bottom: 22px; left: 50%; transform: translateX(-50%) translateY(30px); background: var(--text);
    color: var(--bg); padding: 10px 18px; border-radius: 10px; opacity: 0; transition: all .3s; pointer-events: none; font-size: .9rem; }
  .toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
  .note { color: var(--muted); font-size: .76rem; margin-top: 14px; text-align: center; }
  .spin { display: inline-block; width: 14px; height: 14px; border: 2px solid #fff; border-top-color: transparent;
    border-radius: 50%; animation: sp .7s linear infinite; vertical-align: -2px; margin-right: 8px; }
  @keyframes sp { to { transform: rotate(360deg); } }
  .fade { animation: fade .5s ease; }
  @keyframes fade { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">
      <div class="logo">🎯</div>
      <div>
        <h1>{{ title }}</h1>
        <p class="sub">{{ subtitle }}</p>
      </div>
    </div>
    <button class="icon-btn" id="themeBtn" type="button" aria-label="Toggle theme">🌙 Dark</button>
  </header>

  <div class="grid">
    <!-- FORM -->
    <form id="form" autocomplete="off">
      <div class="card">
        <h2>⚡ Quick fill</h2>
        <div class="presets">
          {% for name in presets %}
          <button type="button" class="chip" data-preset="{{ name }}">{{ name }}</button>
          {% endfor %}
          <button type="button" class="chip" id="randomBtn">🎲 Random</button>
        </div>

        {% for title, icon, fields in groups %}
        <h2 style="margin-top:6px">{{ icon }} {{ title }}</h2>
        <div class="fields" style="margin-bottom:18px">
          {% for f in fields %}
          <div>
            <label for="{{ f.key }}">{{ f.label }} {% if f.hint %}<span class="hint">· {{ f.hint }}</span>{% endif %}</label>
            {% if f.kind == "select" %}
              <select id="{{ f.key }}" name="{{ f.key }}">
                {% for v, t in f.options %}<option value="{{ v }}" {% if v == f.default %}selected{% endif %}>{{ t }}</option>{% endfor %}
              </select>
            {% elif f.kind == "range" %}
              <div class="range-row">
                <input type="range" id="{{ f.key }}" name="{{ f.key }}" min="{{ f.min }}" max="{{ f.max }}" step="{{ f.step }}" value="{{ f.default }}">
                <span class="val" data-for="{{ f.key }}">{{ f.default }}</span>
              </div>
            {% else %}
              <input type="number" id="{{ f.key }}" name="{{ f.key }}" min="{{ f.min }}" max="{{ f.max }}" step="{{ f.step }}" value="{{ f.default }}" required>
            {% endif %}
          </div>
          {% endfor %}
        </div>
        {% endfor %}

        <div class="actions">
          <button class="btn primary" id="submitBtn" type="submit">🔮 Predict</button>
          <button class="btn ghost" id="resetBtn" type="button">↺ Reset</button>
        </div>
        <div class="error" id="error"></div>
      </div>
    </form>

    <!-- RESULT -->
    <div class="sticky">
      <div class="card" id="resultCard">
        <h2>📊 Prediction result</h2>
        <div id="resultBody" class="empty">
          <div class="big">🧠</div>
          <p>Fill in the details and press <b>Predict</b> to see the result here.</p>
        </div>
      </div>
      <div class="card" id="histCard" style="display:none">
        <h2>🕘 Recent predictions</h2>
        <ul class="hist" id="hist"></ul>
      </div>
      <p class="note">Predictions are estimates from a machine learning model and should not be the only basis for a decision.</p>
    </div>
  </div>
</div>
<div class="toast" id="toast"></div>

<script>
  var PRESETS = {{ presets|tojson }};
  var $ = function (s) { return document.querySelector(s); };
  var form = $("#form"), history = [], lastSummary = "";

  // ---------- Theme ----------
  function applyThemeLabel() {
    var dark = document.documentElement.getAttribute("data-theme") === "dark";
    $("#themeBtn").textContent = dark ? "☀️ Light" : "🌙 Dark";
  }
  $("#themeBtn").addEventListener("click", function () {
    var next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("theme", next); } catch (e) {}
    applyThemeLabel();
  });
  applyThemeLabel();

  // ---------- Sliders ----------
  function syncSliders() {
    document.querySelectorAll("input[type=range]").forEach(function (r) {
      var out = document.querySelector('.val[data-for="' + r.id + '"]');
      if (out) out.textContent = r.value;
    });
  }
  document.querySelectorAll("input[type=range]").forEach(function (r) { r.addEventListener("input", syncSliders); });

  // ---------- Presets / reset / random ----------
  function fill(values) {
    Object.keys(values).forEach(function (k) {
      var el = document.getElementById(k);
      if (el) el.value = values[k];
    });
    syncSliders();
  }
  document.querySelectorAll("[data-preset]").forEach(function (b) {
    b.addEventListener("click", function () { fill(PRESETS[b.dataset.preset]); toast("Loaded: " + b.dataset.preset); });
  });
  $("#resetBtn").addEventListener("click", function () {
    form.reset(); syncSliders(); showEmpty(); $("#error").style.display = "none";
  });
  $("#randomBtn").addEventListener("click", function () {
    form.querySelectorAll("input[type=number], input[type=range]").forEach(function (el) {
      var lo = parseFloat(el.min), hi = parseFloat(el.max), st = parseFloat(el.step) || 1;
      var v = lo + Math.random() * (hi - lo);
      v = Math.round(v / st) * st;
      el.value = Math.min(hi, Math.max(lo, +v.toFixed(2)));
    });
    syncSliders(); toast("Random values filled");
  });

  // ---------- Helpers ----------
  function toast(t) {
    var el = $("#toast"); el.textContent = t; el.classList.add("show");
    setTimeout(function () { el.classList.remove("show"); }, 1800);
  }
  function showEmpty() {
    $("#resultBody").className = "empty";
    $("#resultBody").innerHTML = '<div class="big">🧠</div><p>Fill in the details and press <b>Predict</b> to see the result here.</p>';
  }
  function esc(s) { var d = document.createElement("div"); d.textContent = s; return d.innerHTML; }
  function tags(arr, cls) {
    return arr.length ? arr.map(function (t) { return '<span class="tag ' + cls + '">' + esc(t) + "</span>"; }).join("") : '<span class="hint">Nothing to show</span>';
  }
  function color(tone) {
    return getComputedStyle(document.documentElement).getPropertyValue("--" + tone).trim();
  }

  // ---------- Render result ----------
  function render(r) {
    var C = 2 * Math.PI * 54;
    $("#resultBody").className = "fade";
    $("#resultBody").innerHTML =
      '<div class="gauge">' +
        '<svg width="170" height="170" viewBox="0 0 120 120">' +
          '<circle class="track" cx="60" cy="60" r="54" fill="none" stroke-width="10"/>' +
          '<circle class="bar" id="bar" cx="60" cy="60" r="54" fill="none" stroke-width="10" stroke-linecap="round" ' +
            'stroke-dasharray="' + C + '" stroke-dashoffset="' + C + '"/>' +
        "</svg>" +
        '<div class="center"><div><div class="pct" id="pct">0%</div><div class="lbl">chance</div></div></div>' +
      "</div>" +
      '<div class="center-t"><span class="badge tone-' + r.tone + '">' + esc(r.band) + "</span></div>" +
      '<p class="msg">' + esc(r.message) + "</p>" +
      '<div class="list-title">💪 Strengths</div><div class="tags">' + tags(r.strengths, "good") + "</div>" +
      '<div class="list-title">📈 Areas to improve</div><div class="tags">' + tags(r.improve, "imp") + "</div>" +
      '<div class="actions"><button type="button" class="btn ghost" id="copyBtn">📋 Copy result</button></div>';

    var bar = $("#bar");
    bar.style.stroke = color(r.tone === "bad" ? "bad" : r.tone);
    requestAnimationFrame(function () { bar.style.strokeDashoffset = C * (1 - r.probability); });

    // count-up animation
    var start = null, target = r.percent;
    function step(ts) {
      if (!start) start = ts;
      var p = Math.min((ts - start) / 900, 1);
      $("#pct").textContent = Math.round(target * p) + "%";
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);

    lastSummary = r.message + " Strengths: " + (r.strengths.join(", ") || "-") + ". To improve: " + (r.improve.join(", ") || "-") + ".";
    $("#copyBtn").addEventListener("click", function () {
      navigator.clipboard.writeText(lastSummary).then(function () { toast("Result copied"); });
    });

    history.unshift({ band: r.band, percent: r.percent, time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) });
    history = history.slice(0, 5);
    $("#histCard").style.display = "block";
    $("#hist").innerHTML = history.map(function (h) {
      return "<li><span>" + esc(h.band) + " · <b>" + h.percent + "%</b></span><span class='hint'>" + h.time + "</span></li>";
    }).join("");
  }

  // ---------- Submit ----------
  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var err = $("#error"); err.style.display = "none";
    var payload = {};
    new FormData(form).forEach(function (v, k) { payload[k] = v; });

    var btn = $("#submitBtn"); btn.disabled = true;
    btn.innerHTML = '<span class="spin"></span>Predicting...';

    fetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) })
      .then(function (res) { return res.json(); })
      .then(function (r) {
        if (!r.ok) { err.textContent = r.error; err.style.display = "block"; return; }
        render(r);
        if (window.innerWidth <= 920) $("#resultCard").scrollIntoView({ behavior: "smooth" });
      })
      .catch(function () { err.textContent = "Network error. Please try again."; err.style.display = "block"; })
      .finally(function () { btn.disabled = false; btn.innerHTML = "🔮 Predict"; });
  });

  syncSliders();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)