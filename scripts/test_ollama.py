"""
test_ollama.py
--------------
Quick connectivity test for the LLM server at http://172.22.132.20:8001

Tests:
  1. GET  /health      — server + Ollama reachable
  2. POST /read-label  — vision inference with a synthetic image

Run:
  python scripts/test_ollama.py
"""

import base64
import sys

import numpy as np
import cv2
import requests

BASE_URL = "http://172.22.132.20:8001"


def test_health():
    print("[1/2] GET /health ...")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        r.raise_for_status()
        print(f"      Response: {r.json()}")
        print("      PASS\n")
        return True
    except Exception as e:
        print(f"      FAIL: {e}\n")
        return False


def test_read_label():
    print("[2/2] POST /read-label (synthetic image) ...")
    # 64x64 red square with white text
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = (0, 0, 200)  # BGR red
    cv2.putText(img, "TEST", (4, 44), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    _, buf = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buf).decode("utf-8")

    try:
        r = requests.post(f"{BASE_URL}/read-label",
                          json={"image_b64": b64}, timeout=60)
        r.raise_for_status()
        data = r.json()
        print(f"      label_text : {data.get('label_text')!r}")
        print(f"      confidence : {data.get('confidence')}")
        print(f"      latency_s  : {data.get('latency_s')}")
        print("      PASS\n")
        return True
    except Exception as e:
        print(f"      FAIL: {e}\n")
        return False


if __name__ == "__main__":
    print(f"Target: {BASE_URL}\n")

    ok1 = test_health()
    ok2 = test_read_label()

    if ok1 and ok2:
        print("All tests passed — LLM server is reachable and vision works.")
        sys.exit(0)
    else:
        print("One or more tests failed.")
        sys.exit(1)
