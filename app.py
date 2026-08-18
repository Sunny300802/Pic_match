

import argparse
import ipaddress
import os
import json
from flask import Flask, request, render_template_string, send_from_directory, jsonify, send_file, abort
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
import io

import face_engine as fe
from fix_orientation import EXIF_ORIENTATION_TAG, pil_to_bgr, rotate_pil_clockwise


GROUP_PHOTO_FOLDER = "group_photos"
CACHE_FILE = "embeddings_cache.json"
IMAGE_EXTS = fe.IMAGE_EXTS
# ========================================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 48 * 1024 * 1024  

model_error = None
group_cache = {}   
embedding_cache = fe.EmbeddingCache(CACHE_FILE)
GALLERY_INDEX_FILE = "gallery_index.json"
gallery_index = {}  


def ensure_models():
    global model_error
    try:
        fe.ensure_models()
        model_error = None
        return True
    except fe.ModelError as exc:
        model_error = str(exc)
        return False


def index_photo(fname):
    path = os.path.join(GROUP_PHOTO_FOLDER, fname)
    embeds = embedding_cache.get(fname, path)
    if embeds:
        group_cache[fname] = embeds
        return True
    group_cache.pop(fname, None)
    return False


def build_group_cache():
    global model_error
    group_cache.clear()
    if not os.path.isdir(GROUP_PHOTO_FOLDER):
        model_error = f"Folder '{GROUP_PHOTO_FOLDER}' doesn't exist. Edit GROUP_PHOTO_FOLDER at the top of app.py."
        return
    names = fe.list_images(GROUP_PHOTO_FOLDER)
    for fname in names:
        index_photo(fname)
    embedding_cache.prune(names)
    embedding_cache.save()


def load_gallery_index():
    global gallery_index
    try:
        if os.path.isfile(GALLERY_INDEX_FILE):
            with open(GALLERY_INDEX_FILE, "r", encoding="utf-8") as f:
                gallery_index = json.load(f)
        else:
            gallery_index = {}
    except Exception:
        gallery_index = {}


def reference_embeddings(files):
    people, skipped = {}, []
    for f in files:
        name = os.path.splitext(f.filename)[0]
        embedding = fe.embed_primary_face(f.read())
        if embedding is None:
            skipped.append(f.filename)
            continue
        people[name] = embedding
    return people, skipped


def find_matches(people, require_all):
    results = []
    for fname, group_embeds in group_cache.items():
        matched = [name for name, ref in people.items()
                   if fe.is_match(fe.best_similarity(ref, group_embeds))]
        if not matched:
            continue
        if require_all and len(matched) < len(people):
            continue
        results.append((fname, sorted(matched)))
    # photos with more of the uploaded people together float to the top
    results.sort(key=lambda r: (-len(r[1]), r[0]))
    return results


def unique_path(folder, filename):
    """Return a save path that won't overwrite an existing file, appending
    _1, _2, ... to the name if needed."""
    base, ext = os.path.splitext(filename)
    candidate = filename
    i = 1
    while os.path.isfile(os.path.join(folder, candidate)):
        candidate = f"{base}_{i}{ext}"
        i += 1
    return os.path.join(folder, candidate)

def get_uploaded_files(field_name="photos"):

    app.logger.debug("request.files keys: %s", list(request.files.keys()))
    files = [f for f in request.files.getlist(field_name) if f and f.filename]
    if not files:
        # fallback to any uploaded files (useful for single-file uploads
        # where clients use a different field name)
        files = [f for f in request.files.values() if f and f.filename]
    return files


def count_group_photos():
    if not os.path.isdir(GROUP_PHOTO_FOLDER):
        return 0
    return len([f for f in os.listdir(GROUP_PHOTO_FOLDER) if f.lower().endswith(IMAGE_EXTS)])


# ---------------------------------------------------------------- templates
INDEX_HTML = """
<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Light Table &mdash; Photo Match</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
  :root{--bg:#EAE7E0;--surface:#FFFFFF;--ink:#211D19;--ink-soft:#706A5E;--accent:#C81D25;--accent-soft:rgba(200,29,37,.08);--line:#D8D3C7}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:'Inter',sans-serif;min-height:100vh}
  .wrap{max-width:640px;margin:0 auto;padding:64px 24px 80px}
  .eyebrow{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.12em;color:var(--ink-soft);text-transform:uppercase;margin:0 0 14px}
  .eyebrow::before{content:"\\25CF ";color:var(--accent)}
  h1{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:clamp(32px,5vw,44px);line-height:1.05;margin:0 0 12px;letter-spacing:-.01em}
  .sub{color:var(--ink-soft);font-size:16px;line-height:1.5;margin:0 0 40px;max-width:46ch}
  .dropzone{position:relative;border:2px dashed var(--line);border-radius:4px;background:var(--surface);padding:48px 24px;text-align:center;transition:border-color .15s ease,background .15s ease}
  .dropzone.drag{border-color:var(--accent);background:var(--accent-soft)}
  .dropzone p{margin:0 0 4px;font-size:15px}
  .dropzone .hint{color:var(--ink-soft);font-size:13px}
  .dropzone input[type=file]{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:pointer}
  #filename{font-family:'IBM Plex Mono',monospace;font-size:13px;color:var(--accent);margin-top:14px;min-height:18px}
  button{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:15px;background:var(--ink);color:var(--surface);border:none;padding:14px 28px;border-radius:4px;cursor:pointer;margin-top:24px;width:100%}
  button:hover{background:var(--accent)}
  button:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
  button:disabled{opacity:.5;cursor:not-allowed}
  .status{display:flex;align-items:center;gap:8px;font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--ink-soft);margin-top:28px;letter-spacing:.02em}
  .status .count{color:var(--ink);font-weight:600}
  .rescan-row{margin-top:6px}
  .rescan-row button{all:unset;cursor:pointer;font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--ink-soft);text-decoration:underline;text-underline-offset:3px}
  .rescan-row button:hover{color:var(--accent)}
  .banner{font-family:'IBM Plex Mono',monospace;font-size:13px;padding:12px 16px;border-radius:4px;margin-bottom:24px}
  .banner.error{background:var(--accent-soft);color:var(--accent);border:1px solid var(--accent)}
  .banner.ok{background:#eef4ec;color:#2f5e3f;border:1px solid #b9d3bd}
  .check-row{display:flex;align-items:center;gap:9px;margin-top:18px;font-size:14px;color:var(--ink-soft);cursor:pointer;user-select:none}
  .check-row input{width:16px;height:16px;accent-color:var(--accent);cursor:pointer}
</style></head><body>
<div class="wrap">
  <p class="eyebrow">Light Table / Photo Match</p>
  <h1>Find the frames<br>they're in.</h1>
  <p class="sub">Upload one photo per person &mdash; select several at once to check for a whole group in the same shot.</p>

  {% if error %}<div class="banner error">{{ error }}</div>{% endif %}
  {% if message %}<div class="banner ok">{{ message }}</div>{% endif %}

  <form action="/match" method="post" enctype="multipart/form-data" id="uploadForm">
    <div class="dropzone" id="dropzone">
      <p>Drop photos here, or click to browse</p>
      <p class="hint">JPG, PNG, or WEBP &middot; select multiple to check several people</p>
      <input type="file" name="photos" accept="image/*" id="fileInput" required multiple>
    </div>
    <p id="filename"></p>
    <label class="check-row"><input type="checkbox" name="require_all"> <span>Only show photos with everyone together</span></label>
    <button type="submit" id="submitBtn" disabled>Find matches</button>
  </form>

  <div class="status"><span>&#9670; indexed:</span> <span class="count">{{ num_cached }}</span> <span>group photo(s)</span></div>
  <div class="rescan-row"><form action="/rescan" method="post"><button type="submit">rescan group_photos folder</button></form></div>
  <div class="rescan-row"><a href="/admin" style="color:var(--ink-soft);text-decoration:underline;text-underline-offset:3px">+ add group photos</a></div>
</div>
<script>
  const dz = document.getElementById('dropzone');
  const input = document.getElementById('fileInput');
  const nameEl = document.getElementById('filename');
  const btn = document.getElementById('submitBtn');
  function showNames(files){
    if(!files || !files.length) return;
    if(files.length === 1){ nameEl.textContent = '\\u2192 ' + files[0].name; }
    else { nameEl.textContent = '\\u2192 ' + files.length + ' photos: ' + Array.from(files).map(f => f.name).join(', '); }
    btn.disabled = false;
  }
  input.addEventListener('change', () => showNames(input.files));
  ['dragenter','dragover'].forEach(evt => dz.addEventListener(evt, e => { e.preventDefault(); dz.classList.add('drag'); }));
  ['dragleave','drop'].forEach(evt => dz.addEventListener(evt, e => { e.preventDefault(); dz.classList.remove('drag'); }));
  dz.addEventListener('drop', e => { if(e.dataTransfer.files.length){ input.files = e.dataTransfer.files; showNames(input.files); } });
</script>
</body></html>
"""

ADMIN_HTML = """
<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Add photos &mdash; Light Table</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
  :root{--bg:#EAE7E0;--surface:#FFFFFF;--ink:#211D19;--ink-soft:#706A5E;--accent:#C81D25;--accent-soft:rgba(200,29,37,.08);--line:#D8D3C7}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:'Inter',sans-serif;min-height:100vh}
  .wrap{max-width:640px;margin:0 auto;padding:64px 24px 80px}
  .eyebrow{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.12em;color:var(--ink-soft);text-transform:uppercase;margin:0 0 14px}
  .eyebrow::before{content:"\\25CF ";color:var(--accent)}
  h1{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:clamp(32px,5vw,44px);line-height:1.05;margin:0 0 12px;letter-spacing:-.01em}
  .sub{color:var(--ink-soft);font-size:16px;line-height:1.5;margin:0 0 40px;max-width:46ch}
  .dropzone{position:relative;border:2px dashed var(--line);border-radius:4px;background:var(--surface);padding:48px 24px;text-align:center;transition:border-color .15s ease,background .15s ease}
  .dropzone.drag{border-color:var(--accent);background:var(--accent-soft)}
  .dropzone p{margin:0 0 4px;font-size:15px}
  .dropzone .hint{color:var(--ink-soft);font-size:13px}
  .dropzone input[type=file]{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:pointer}
  #filename{font-family:'IBM Plex Mono',monospace;font-size:13px;color:var(--accent);margin-top:14px;min-height:18px}
  button{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:15px;background:var(--ink);color:var(--surface);border:none;padding:14px 28px;border-radius:4px;cursor:pointer;margin-top:24px;width:100%}
  button:hover{background:var(--accent)}
  button:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
  button:disabled{opacity:.5;cursor:not-allowed}
  .status{display:flex;align-items:center;gap:8px;font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--ink-soft);margin-top:28px;letter-spacing:.02em}
  .status .count{color:var(--ink);font-weight:600}
  .banner{font-family:'IBM Plex Mono',monospace;font-size:13px;padding:12px 16px;border-radius:4px;margin-bottom:24px}
  .banner.error{background:var(--accent-soft);color:var(--accent);border:1px solid var(--accent)}
  .banner.ok{background:#eef4ec;color:#2f5e3f;border:1px solid #b9d3bd}
  a.back{display:inline-block;margin-top:28px;color:var(--ink-soft);font-family:'IBM Plex Mono',monospace;font-size:13px;text-decoration:none;border-bottom:1px solid var(--line);padding-bottom:2px}
  a.back:hover{color:var(--accent);border-color:var(--accent)}
</style></head><body>
<div class="wrap">
  <p class="eyebrow">Light Table / Admin</p>
  <h1>Add group photos</h1>
  <p class="sub">Upload new pictures straight into your group_photos library. They're indexed right away, so people can be matched against them immediately &mdash; no separate rescan needed.</p>

  {% if error %}<div class="banner error">{{ error }}</div>{% endif %}
  {% if message %}<div class="banner ok">{{ message }}</div>{% endif %}

  <form action="/admin/upload" method="post" enctype="multipart/form-data" id="uploadForm">
    <div class="dropzone" id="dropzone">
      <p>Drop group photos here, or click to browse</p>
      <p class="hint">JPG, PNG, or WEBP &middot; select as many as you like</p>
      <input type="file" name="photos" accept="image/*" id="fileInput" required multiple>
    </div>
    <p id="filename"></p>
    <button type="submit" id="submitBtn" disabled>Upload to library</button>
  </form>

  <div class="status"><span>&#9670; on disk:</span> <span class="count">{{ total_files }}</span> <span>photo(s), {{ num_cached }} indexed</span></div>
  <a class="back" href="/">&larr; back to search</a>
</div>
<script>
  const dz = document.getElementById('dropzone');
  const input = document.getElementById('fileInput');
  const nameEl = document.getElementById('filename');
  const btn = document.getElementById('submitBtn');
  function showNames(files){
    if(!files || !files.length) return;
    if(files.length === 1){ nameEl.textContent = '\\u2192 ' + files[0].name; }
    else { nameEl.textContent = '\\u2192 ' + files.length + ' photos: ' + Array.from(files).map(f => f.name).join(', '); }
    btn.disabled = false;
  }
  input.addEventListener('change', () => showNames(input.files));
  ['dragenter','dragover'].forEach(evt => dz.addEventListener(evt, e => { e.preventDefault(); dz.classList.add('drag'); }));
  ['dragleave','drop'].forEach(evt => dz.addEventListener(evt, e => { e.preventDefault(); dz.classList.remove('drag'); }));
  dz.addEventListener('drop', e => { if(e.dataTransfer.files.length){ input.files = e.dataTransfer.files; showNames(input.files); } });
</script>
</body></html>
"""

RESULTS_HTML = """
<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Matches &mdash; Light Table</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
  :root{--bg:#EAE7E0;--surface:#FFFFFF;--ink:#211D19;--ink-soft:#706A5E;--accent:#C81D25;--accent-soft:rgba(200,29,37,.08);--line:#D8D3C7}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:'Inter',sans-serif}
  .wrap{max-width:1040px;margin:0 auto;padding:56px 24px 80px}
  .eyebrow{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.12em;color:var(--ink-soft);text-transform:uppercase;margin:0 0 14px}
  .eyebrow::before{content:"\\25CF ";color:var(--accent)}
  h1{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:clamp(26px,4vw,36px);margin:0 0 8px;letter-spacing:-.01em}
  .meta{color:var(--ink-soft);font-size:14px;margin:0 0 6px;font-family:'IBM Plex Mono',monospace}
  .meta b{color:var(--ink)}
  .meta.warn{color:var(--accent)}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:26px;margin-top:34px}

  .frame{background:var(--surface);border-radius:4px;border:1px solid var(--line);position:relative;opacity:0;animation:rise .4s ease forwards;cursor:pointer}
  .frame:hover{border-color:var(--ink-soft)}
  .frame:focus-visible{outline:3px solid var(--accent);outline-offset:3px}
  .frame:nth-child(1){animation-delay:.02s}.frame:nth-child(2){animation-delay:.06s}.frame:nth-child(3){animation-delay:.10s}
  .frame:nth-child(4){animation-delay:.14s}.frame:nth-child(5){animation-delay:.18s}.frame:nth-child(6){animation-delay:.22s}
  .frame:nth-child(n+7){animation-delay:.24s}
  @keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}

  .thumb{overflow:hidden;border-radius:4px 4px 0 0;position:relative}
  .thumb img{width:100%;height:200px;object-fit:cover;display:block;transition:transform .2s ease}
  .frame:hover .thumb img{transform:scale(1.04)}
  .expand-hint{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(33,29,25,0);transition:background .15s ease}
  .frame:hover .expand-hint{background:rgba(33,29,25,.15)}
  .expand-hint svg{opacity:0;transition:opacity .15s ease}
  .frame:hover .expand-hint svg{opacity:1}

  .stamp{padding:10px 12px 12px;font-family:'IBM Plex Mono',monospace;border-top:1px solid var(--line);border-radius:0 0 4px 4px}
  .stamp .fname{display:block;font-size:12px;color:var(--ink);font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-bottom:8px}
  .tags{display:flex;flex-wrap:wrap;gap:5px}
  .tag{background:var(--accent-soft);color:var(--accent);padding:2px 9px;border-radius:20px;font-size:11px}
  .tag.tag-full{background:var(--accent);color:#fff}

  .empty{border:2px dashed var(--line);border-radius:4px;padding:64px 24px;text-align:center;color:var(--ink-soft);font-family:'IBM Plex Mono',monospace;font-size:14px;margin-top:34px}
  a.back{display:inline-block;margin-top:44px;color:var(--ink-soft);font-family:'IBM Plex Mono',monospace;font-size:13px;text-decoration:none;border-bottom:1px solid var(--line);padding-bottom:2px}
  a.back:hover{color:var(--accent);border-color:var(--accent)}
  a.back:focus-visible{outline:3px solid var(--accent);outline-offset:4px}
  @media (prefers-reduced-motion:reduce){.frame{opacity:1;animation:none}.thumb img{transition:none}}

  /* ---- lightbox ---- */
  .lightbox{display:none;position:fixed;inset:0;background:rgba(20,17,15,.94);z-index:1000;flex-direction:column}
  .lightbox.open{display:flex}
  .lb-toolbar{display:flex;align-items:center;gap:12px;padding:14px 20px;font-family:'IBM Plex Mono',monospace;color:#EAE7E0;font-size:13px}
  .lb-toolbar button,.lb-toolbar a{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.25);color:#EAE7E0;width:36px;height:36px;border-radius:4px;display:flex;align-items:center;justify-content:center;cursor:pointer;text-decoration:none;font-size:19px;line-height:1}
  .lb-toolbar button:hover,.lb-toolbar a:hover{background:var(--accent);border-color:var(--accent)}
  .lb-toolbar button:focus-visible,.lb-toolbar a:focus-visible{outline:2px solid #fff;outline-offset:2px}
  #lbZoomLevel{min-width:44px;text-align:center}
  #lbClose{margin-left:auto;font-size:22px}
  .lb-stage{flex:1;overflow:hidden;display:flex;align-items:center;justify-content:center;cursor:grab}
  .lb-stage.dragging{cursor:grabbing}
  .lb-stage img{max-width:90vw;max-height:80vh;user-select:none;-webkit-user-drag:none}
  .lb-caption{text-align:center;color:var(--ink-soft);font-family:'IBM Plex Mono',monospace;font-size:12px;padding:6px 0 18px}
</style></head><body>
<div class="wrap">
  <p class="eyebrow">Light Table / Photo Match</p>
  <h1>Matches</h1>
  <p class="meta">Checking for: <b>{{ people|join(', ') }}</b></p>
  {% if skipped %}<p class="meta warn">Couldn't find a face in: {{ skipped|join(', ') }} &mdash; skipped</p>{% endif %}
  {% if require_all %}<p class="meta">Showing only photos with everyone together</p>{% endif %}

  {% if results %}
    <p class="meta">{{ results|length }} group photo(s), sorted by how many of them are in each</p>
    <div class="grid">
      {% for fname, names in results %}
      <div class="frame" data-full="/group_photo/{{ fname }}" data-name="{{ fname }}" tabindex="0" role="button" aria-label="View {{ fname }} full size">
        <div class="thumb">
          <img src="/group_photo/{{ fname }}" alt="{{ fname }}" loading="lazy">
          <div class="expand-hint"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2"><path d="M9 3H5a2 2 0 0 0-2 2v4M15 3h4a2 2 0 0 1 2 2v4M9 21H5a2 2 0 0 1-2-2v-4M15 21h4a2 2 0 0 0 2-2v-4"/></svg></div>
        </div>
        <div class="stamp">
          <span class="fname">{{ fname }}</span>
          <span class="tags">
            {% for n in names %}<span class="tag{% if names|length == people|length %} tag-full{% endif %}">{{ n }}</span>{% endfor %}
          </span>
        </div>
      </div>
      {% endfor %}
    </div>
  {% else %}
    <div class="empty">No group photo matches {{ 'all of them together' if require_all else 'any of them' }}.<br>Try clearer, front-facing reference photos{% if require_all %}, or untick "everyone together"{% endif %}.</div>
  {% endif %}
  <a class="back" href="/">&larr; try again</a>
</div>

<div class="lightbox" id="lightbox">
  <div class="lb-toolbar">
    <button id="lbZoomOut" aria-label="Zoom out">&minus;</button>
    <span id="lbZoomLevel">100%</span>
    <button id="lbZoomIn" aria-label="Zoom in">+</button>
    <a id="lbDownload" aria-label="Download this photo">&#8595;</a>
    <button id="lbClose" aria-label="Close">&times;</button>
  </div>
  <div class="lb-stage" id="lbStage"><img id="lbImg" src="" alt=""></div>
  <p class="lb-caption" id="lbCaption"></p>
</div>

<script>
(function(){
  const lightbox = document.getElementById('lightbox');
  const lbImg = document.getElementById('lbImg');
  const lbStage = document.getElementById('lbStage');
  const zoomLevelEl = document.getElementById('lbZoomLevel');
  const lbDownload = document.getElementById('lbDownload');
  const lbCaption = document.getElementById('lbCaption');
  let scale = 1, tx = 0, ty = 0, dragging = false, startX = 0, startY = 0;

  function applyTransform(){
    lbImg.style.transform = 'translate(' + tx + 'px, ' + ty + 'px) scale(' + scale + ')';
    zoomLevelEl.textContent = Math.round(scale * 100) + '%';
  }
  function resetView(){ scale = 1; tx = 0; ty = 0; applyTransform(); }
  function zoomBy(delta){
    scale = Math.min(Math.max(scale + delta, 1), 4);
    if (scale === 1) { tx = 0; ty = 0; }
    applyTransform();
  }
  function openLightbox(frame){
    lbImg.src = frame.dataset.full;
    lbCaption.textContent = frame.dataset.name;
    lbDownload.href = frame.dataset.full;
    lbDownload.setAttribute('download', frame.dataset.name);
    resetView();
    lightbox.classList.add('open');
  }
  function closeLightbox(){ lightbox.classList.remove('open'); }

  document.querySelectorAll('.frame').forEach(frame => {
    frame.addEventListener('click', () => openLightbox(frame));
    frame.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openLightbox(frame); }
    });
  });

  document.getElementById('lbZoomIn').addEventListener('click', () => zoomBy(0.25));
  document.getElementById('lbZoomOut').addEventListener('click', () => zoomBy(-0.25));
  document.getElementById('lbClose').addEventListener('click', closeLightbox);
  lightbox.addEventListener('click', e => { if (e.target === lightbox) closeLightbox(); });

  lbStage.addEventListener('wheel', e => {
    e.preventDefault();
    zoomBy(e.deltaY < 0 ? 0.15 : -0.15);
  }, { passive: false });

  lbStage.addEventListener('mousedown', e => {
    if (scale <= 1) return;
    dragging = true; lbStage.classList.add('dragging');
    startX = e.clientX - tx; startY = e.clientY - ty;
  });
  window.addEventListener('mousemove', e => {
    if (!dragging) return;
    tx = e.clientX - startX; ty = e.clientY - startY;
    applyTransform();
  });
  window.addEventListener('mouseup', () => { dragging = false; lbStage.classList.remove('dragging'); });

  window.addEventListener('keydown', e => {
    if (!lightbox.classList.contains('open')) return;
    if (e.key === 'Escape') closeLightbox();
    if (e.key === '+' || e.key === '=') zoomBy(0.25);
    if (e.key === '-') zoomBy(-0.25);
  });
})();
</script>
</body></html>
"""


# --------------------------------------------------------------------- routes
@app.route("/")
def index():
    err = model_error if not ensure_models() else None
    return render_template_string(INDEX_HTML, num_cached=len(group_cache), error=err, message=None)


@app.route("/rescan", methods=["POST"])
def rescan():
    if ensure_models():
        build_group_cache()
        load_gallery_index()
        msg = f"Rescanned - {len(group_cache)} group photo(s) with a detected face are now indexed."
        err = model_error
    else:
        msg, err = None, model_error
    return render_template_string(INDEX_HTML, num_cached=len(group_cache), error=err, message=msg)


@app.route("/admin")
def admin():
    return render_template_string(ADMIN_HTML, num_cached=len(group_cache),
                                   total_files=count_group_photos(), error=None, message=None)


@app.route("/admin/upload", methods=["POST"])
def admin_upload():
    os.makedirs(GROUP_PHOTO_FOLDER, exist_ok=True)

    files = [f for f in request.files.getlist("photos") if f and f.filename]
    if not files:
        return render_template_string(ADMIN_HTML, num_cached=len(group_cache),
                                       total_files=count_group_photos(),
                                       error="Choose at least one photo to upload.", message=None)

    saved, rejected, straightened = [], [], []
    for f in files:
      safe_name = secure_filename(f.filename)
      if not safe_name or not safe_name.lower().endswith(IMAGE_EXTS):
        rejected.append(f.filename)
        continue
      dest = unique_path(GROUP_PHOTO_FOLDER, safe_name)
   
      try:
        img_pil = ImageOps.exif_transpose(Image.open(f.stream))
        turn = 0
        if ensure_models():
          turn, _ = fe.detect_upright_rotation(pil_to_bgr(img_pil))
        if turn:
          img_pil = rotate_pil_clockwise(img_pil, turn)
          straightened.append(f"{os.path.basename(dest)} ({turn}°)")
        exif = img_pil.getexif()
        exif.pop(EXIF_ORIENTATION_TAG, None)  
        save_kwargs = {"exif": exif.tobytes()} if len(exif) else {}
        img_pil.convert("RGB").save(dest, **save_kwargs)
      except Exception:
   
        app.logger.exception("could not re-save %s upright; storing it as uploaded", safe_name)
        straightened = [s for s in straightened if not s.startswith(os.path.basename(dest))]
        f.stream.seek(0)
        with open(dest, "wb") as out:
          out.write(f.stream.read())
      saved.append(os.path.basename(dest))

    indexed = 0
    if saved and ensure_models():
        for fname in saved:
            if index_photo(fname):
                indexed += 1
        embedding_cache.save()

    parts = []
    if saved:
        parts.append(f"Added {len(saved)} photo(s): {', '.join(saved)} ({indexed} indexed, "
                      f"{len(saved) - indexed} had no detectable face but are still saved)")
    if straightened:
        parts.append(f"Straightened sideways photo(s): {', '.join(straightened)}")
    if rejected:
        parts.append(f"Skipped (not an image file): {', '.join(rejected)}")

    return render_template_string(
        ADMIN_HTML, num_cached=len(group_cache), total_files=count_group_photos(),
        message=" · ".join(parts) if saved else None,
        error=None if saved else " · ".join(parts) if parts else "Nothing was uploaded.",
    )


@app.route("/match", methods=["POST"])
def match():
    if not ensure_models():
        return render_template_string(INDEX_HTML, num_cached=len(group_cache), error=model_error, message=None)

    if not group_cache:
        build_group_cache()

    files = get_uploaded_files("photos")
    if not files:
      return render_template_string(INDEX_HTML, num_cached=len(group_cache),
                       error="Choose at least one photo first.", message=None)

    require_all = request.form.get("require_all") == "on"

    people, skipped = reference_embeddings(files)
    if not people:
        return render_template_string(INDEX_HTML, num_cached=len(group_cache),
                                       error="No face found in any of those photos - try clearer, front-facing shots.",
                                       message=None)

    results = find_matches(people, require_all)

    return render_template_string(RESULTS_HTML, results=results, people=sorted(people.keys()),
                                   skipped=skipped, require_all=require_all)


@app.route("/api/match", methods=["POST"])
def api_match():
  if not ensure_models():
    return jsonify(error=model_error), 500

  if not group_cache:
    build_group_cache()

  files = get_uploaded_files("photos")
  if not files:
    
    return jsonify(error="Choose at least one photo first.", received_keys=list(request.files.keys())), 400

  require_all = request.form.get("require_all") == "on"

  people, skipped = reference_embeddings(files)
  if not people:
    return jsonify(error="No face found in any of those photos - try clearer, front-facing shots."), 400

  results = []
  for fname, matched_names in find_matches(people, require_all):
    orig_entry = gallery_index.get(fname)
    orig = orig_entry.get("original_url") if isinstance(orig_entry, dict) else None
    results.append({
      "filename": fname,
      "url": f"/group_photo/{fname}",
      "original_url": orig,
      "matched_names": matched_names,
    })

  return jsonify({
    "people": sorted(people.keys()),
    "skipped": skipped,
    "require_all": require_all,
    "results": results,
    "indexed_group_photos": len(group_cache),
  })


@app.route("/group_photo/<path:filename>")
def group_photo(filename):
  path = os.path.join(GROUP_PHOTO_FOLDER, filename)
  if not os.path.isfile(path):
    abort(404)
  try:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    buf = io.BytesIO()
    fmt = img.format or "JPEG"
    img.save(buf, format=fmt)
    buf.seek(0)
    return send_file(buf, mimetype=f"image/{fmt.lower()}")
  except Exception:
    return send_from_directory(GROUP_PHOTO_FOLDER, filename)


DOCS_HTML = """
<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>API Docs &mdash; Light Table</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
  :root{--bg:#EAE7E0;--surface:#FFFFFF;--ink:#211D19;--ink-soft:#706A5E;--accent:#C81D25;--line:#D8D3C7}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:'Inter',sans-serif;min-height:100vh}
  .wrap{max-width:760px;margin:0 auto;padding:64px 24px 80px}
  h1{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:clamp(32px,5vw,44px);margin:0 0 16px}
  .section{margin-top:32px}
  .section h2{font-size:20px;margin:0 0 12px}
  .section p, .section pre, .section ul{margin:0 0 16px;line-height:1.6}
  .card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:22px;}
  pre{background:rgba(33,29,25,.04);padding:16px;border-radius:8px;overflow-x:auto}
  code{font-family:'IBM Plex Mono',monospace;background:rgba(33,29,25,.06);padding:.15em .35em;border-radius:.35em}
  a{color:var(--accent);text-decoration:none}
  a:hover{text-decoration:underline}
</style>
</head><body>
<div class="wrap">
  <h1>Light Table API Docs</h1>
  <p>Quick reference for the built-in routes exposed by this photo matching app.</p>

  <div class="section card">
    <h2>GET /</h2>
    <p>Loads the main search page where you can upload reference photos and find matching group photos.</p>
  </div>

  <div class="section card">
    <h2>POST /match</h2>
    <p>Upload one or more reference photos to search the indexed group photos.</p>
    <p>Form fields:</p>
    <ul>
      <li><code>photos</code> — one or more image files</li>
      <li><code>require_all</code> — optional checkbox; when present, only returns photos containing every uploaded person</li>
    </ul>
  </div>

  <div class="section card">
    <h2>POST /rescan</h2>
    <p>Rebuilds the face embedding index from the images currently stored in <code>group_photos/</code>.</p>
  </div>

  <div class="section card">
    <h2>GET /admin</h2>
    <p>Loads the admin page for uploading new group photos into the app library.</p>
  </div>

  <div class="section card">
    <h2>POST /admin/upload</h2>
    <p>Upload group photos to be saved into <code>group_photos/</code> and indexed automatically.</p>
    <p>Form fields:</p>
    <ul>
      <li><code>photos</code> — one or more image files</li>
    </ul>
  </div>

  <div class="section card">
    <h2>GET /group_photo/&lt;filename&gt;</h2>
    <p>Serves a specific stored group photo by filename.</p>
  </div>

  <div class="section card">
    <h2>POST /api/match</h2>
    <p>Returns a JSON-only result set with group photo matches and image URLs.</p>
    <p>Form fields:</p>
    <ul>
      <li><code>photos</code> — one or more image files</li>
      <li><code>require_all</code> — optional checkbox field; include if you want only photos containing every uploaded person</li>
    </ul>
  </div>

  <div class="section card">
    <h2>GET /docs</h2>
    <p>This page.</p>
  </div>

  <div class="section">
    <h2>Example cURL</h2>
    <pre>curl -F "photos=@person1.jpg" -F "photos=@person2.jpg" http://127.0.0.1:5001/match</pre>
  </div>
</div>
</body></html>
"""


@app.route("/docs")
def docs():
    return render_template_string(DOCS_HTML)


def parse_args():
    parser = argparse.ArgumentParser(description="Photo match web app")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host/IP address to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5001,
                        help="Port to listen on (default: 5001)")
    args = parser.parse_args()

    try:
        ipaddress.ip_address(args.host)
    except ValueError:
        parser.error(f"Invalid host address: {args.host}")

    if not (1 <= args.port <= 65535):
        parser.error("Port must be between 1 and 65535")

    return args


if __name__ == "__main__":
    args = parse_args()

    print("Loading models and scanning group_photos/ ...")
    print(fe.settings_summary())
    if ensure_models():
        build_group_cache()
        load_gallery_index()
        faces = sum(len(v) for v in group_cache.values())
        print(f"Indexed {len(group_cache)} group photo(s), {faces} face(s).")
    else:
        print(f"Warning: {model_error}")
        print("The app will still start - fix the issue above, then click 'rescan' on the page.")

    print(f"\nOpen this address in your browser: http://{args.host}:{args.port}\n")
    app.run(debug=False, port=args.port, host=args.host)
