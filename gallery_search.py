

import os
import csv
import requests
import threading
import time
import json
from flask import Flask, jsonify

import face_engine as fe
from match_faces import SINGLE_PERSON_FOLDER, load_reference_people

API_URL = ""
GROUP_PHOTO_FOLDER = "group_photos"
INDEX_FILE = "gallery_index.json"
CACHE_FILE = "embeddings_cache.json"
FETCH_INTERVAL_SECONDS = 15 * 60  # 15 minutes

app = Flask(__name__)

gallery_index = {}
index_lock = threading.Lock()


def fetch_gallery_json(url):
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()


def download_image(url, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    filename = os.path.basename(url.split("?")[0])
    dest_path = os.path.join(dest_folder, filename)
    if os.path.isfile(dest_path):
        return dest_path
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(1024 * 8):
            if chunk:
                f.write(chunk)
    return dest_path


def update_gallery_index():
    try:
        data = fetch_gallery_json(API_URL)
    except Exception as e:
        print("Failed to fetch gallery JSON:", e)
        return

    if not data or not data.get("status"):
        print("API did not return a successful response:", data)
        return

    image_entries = []
    days = data.get("data", {}).get("days", [])
    for day in days:
        for img in day.get("images", []):
            url = img.get("image_url") or img.get("image_path")
            if url:
                image_entries.append({"original_url": url})

    if not image_entries:
        print("No images found in gallery response.")
        return

    downloaded = []
    for entry in image_entries:
        url = entry["original_url"]
        try:
            path = download_image(url, GROUP_PHOTO_FOLDER)
            fname = os.path.basename(path)
            downloaded.append(fname)
            with index_lock:
                gallery_index[fname] = {"original_url": url, "local_path": path}
            print("  downloaded:", path)
        except Exception as e:
            print(f"  ! failed to download {url}: {e}")

    try:
        with index_lock:
            with open(INDEX_FILE, "w", encoding="utf-8") as f:
                json.dump(gallery_index, f, indent=2)
    except Exception as e:
        print("Failed to write index file:", e)


def periodic_fetch_loop():

    update_gallery_index()
    while True:
        time.sleep(FETCH_INTERVAL_SECONDS)
        update_gallery_index()


@app.route("/gallery", methods=["GET"])
def gallery_endpoint():
    with index_lock:
        return jsonify({"status": True, "images": gallery_index})


def build_and_match_all():
    try:
        fe.ensure_models()
    except fe.ModelError as exc:
        print("Error:", exc)
        return
    print(fe.settings_summary())
    print("Reading reference photos from", SINGLE_PERSON_FOLDER)
    person_embeddings, _ = load_reference_people(SINGLE_PERSON_FOLDER)
    if not person_embeddings:
        print("No reference faces found. Skipping matching.")
        return

    print("\nScanning downloaded group photos...")
    cache = fe.EmbeddingCache(CACHE_FILE)
    results_by_person = {name: [] for name in person_embeddings}
    results_by_photo = {}

    with index_lock:
        photo_names = sorted(gallery_index.keys())

    for fname in photo_names:
        path = os.path.join(GROUP_PHOTO_FOLDER, fname)
        if not os.path.isfile(path):
            continue
        group_embeds = cache.get(fname, path)
        if not group_embeds:
            print(f"  ! no faces detected in {fname}")
            results_by_photo[fname] = []
            continue

        matched_people = []
        for name, ref_embed in person_embeddings.items():
            if fe.is_match(fe.best_similarity(ref_embed, group_embeds)):
                results_by_person[name].append(fname)
                matched_people.append(name)
        results_by_photo[fname] = sorted(matched_people)

    cache.prune(photo_names)
    cache.save()

    with open("results_by_person.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["person", "num_group_photos", "group_photo_filenames"])
        for name, photos in results_by_person.items():
            writer.writerow([name, len(photos), ", ".join(photos)])

    with open("results_by_photo.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group_photo", "people_present"])
        for fname, people in results_by_photo.items():
            writer.writerow([fname, ", ".join(people)])

    print("Saved results_by_person.csv and results_by_photo.csv")


def load_index_from_disk():
    global gallery_index
    if os.path.isfile(INDEX_FILE):
        try:
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                gallery_index = json.load(f)
        except Exception:
            gallery_index = {}


if __name__ == "__main__":
    load_index_from_disk()

    t = threading.Thread(target=periodic_fetch_loop, daemon=True)
    t.start()

    try:
        mm = threading.Thread(target=build_and_match_all, daemon=True)
        mm.start()
    except Exception as e:
        print("Failed to start matching thread:", e)

    print("Starting Flask server on http://127.0.0.1:5000 (endpoint: /gallery)")
    app.run(host="0.0.0.0", port=5000)
