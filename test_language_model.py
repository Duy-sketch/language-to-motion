"""
Test language model sau khi train xong.

Cách dùng:
    python test_language_model.py

Yêu cầu: đã train xong và có folder capsule_language_model/
"""

import numpy as np
import torch
from train_language_model import load_model, predict, ACTION_COLS

# ── Màu terminal ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def color(text, c): return f"{c}{text}{RESET}"

# ── Test cases ────────────────────────────────────────────────────────────────
# (câu lệnh, action đúng, mô tả)
TEST_CASES = [
    # --- Các câu đã có trong training data ---
    ("sang phải",                       [0, 0,  5, 0, 0],  "câu đơn giản"),
    ("sang trái",                       [0, 0, -5, 0, 0],  "câu đơn giản"),
    ("tiến lên",                        [0, 0,  0, 0, 5],  "câu đơn giản"),
    ("lùi lại",                         [0, 0,  0, 0,-5],  "câu đơn giản"),
    ("xoay trái",                       [0, -5, 0, 0, 0],  "câu đơn giản"),
    ("nghiêng xuống",                   [-5, 0, 0, 0, 0],  "câu đơn giản"),
    ("dừng lại",                        [0,  0, 0, 0, 0],  "câu đơn giản"),

    # --- Câu có ngữ cảnh y tế (đã thấy trong training) ---
    ("thấy polyp bên trái, sang phải",  [0, 0,  5, 0, 0],  "câu có ngữ cảnh"),
    ("tiến lại gần tổn thương",         [0, 0,  0, 0, 5],  "câu có ngữ cảnh"),
    ("phát hiện bất thường, nghiêng xuống", [-5, 0, 0, 0, 0], "câu có ngữ cảnh"),

    # --- Câu CHƯA thấy trong training (quan trọng nhất) ---
    ("dịch về bên phải",                [0, 0,  5, 0, 0],  "câu mới ★"),
    ("nhìn lên phía trên",              [5, 0,  0, 0, 0],  "câu mới ★"),
    ("lại gần vị trí đó",               [0, 0,  0, 0, 5],  "câu mới ★"),
    ("bác sĩ muốn xem bên trái",        [0, 0, -5, 0, 0],  "câu mới ★"),
    ("quan sát kỹ hơn phía dưới",       [-5, 0,  0, 0, 0], "câu mới ★"),
]

TOLERANCE = 2.5  # sai số cho phép (mm hoặc độ)

def test_single(cmd, expected, model, tok, lmax):
    result   = predict(cmd, model, tok, lmax)
    got      = result["action"]
    errors   = [abs(g - e) for g, e in zip(got, expected)]
    max_err  = max(errors)
    passed   = max_err <= TOLERANCE
    return passed, got, max_err

def run_tests():
    print("Loading model...")
    model, tok, lmax = load_model()
    print("Model loaded!\n")

    labels = ["Δα°", "Δβ°", "ΔXmm", "ΔYmm", "ΔZmm"]

    passed_total = 0
    results_by_type = {}

    print(f"{'Câu lệnh':<45} {'Type':<22} {'Result':<8} {'Max err'}")
    print("─" * 95)

    for cmd, expected, desc in TEST_CASES:
        passed, got, max_err = test_single(cmd, expected, model, tok, lmax)
        if passed: passed_total += 1

        status = color("PASS ✓", GREEN) if passed else color("FAIL ✗", RED)
        err_str = color(f"{max_err:.1f}", GREEN if passed else RED)

        print(f"{cmd:<45} {desc:<22} {status}  err={err_str}")

        if desc not in results_by_type:
            results_by_type[desc] = {"pass": 0, "total": 0}
        results_by_type[desc]["total"] += 1
        if passed:
            results_by_type[desc]["pass"] += 1

    # ── Tóm tắt ──────────────────────────────────────────────────────────────
    total = len(TEST_CASES)
    pct   = passed_total / total * 100
    print("\n" + "─" * 95)
    print(f"Tổng: {passed_total}/{total} passed ({pct:.0f}%)")
    print()
    for desc, r in results_by_type.items():
        p = r["pass"] / r["total"] * 100
        bar = "█" * r["pass"] + "░" * (r["total"] - r["pass"])
        c = GREEN if p == 100 else (YELLOW if p >= 50 else RED)
        print(f"  {desc:<25} {color(bar, c)}  {r['pass']}/{r['total']}")

    # ── Nhận xét tự động ─────────────────────────────────────────────────────
    print()
    new_cases = [r for desc, r in results_by_type.items() if "mới" in desc]
    new_pct   = new_cases[0]["pass"] / new_cases[0]["total"] * 100 if new_cases else 0

    if pct >= 90 and new_pct >= 70:
        print(color("✓ Model tốt — sẵn sàng test trên phantom!", GREEN))
    elif pct >= 70:
        print(color("△ Model tạm ổn — nên thêm data đa dạng hơn trước khi test thật.", YELLOW))
    else:
        print(color("✗ Model chưa đủ tốt — cần thêm data hoặc train thêm epochs.", RED))

    # ── Interactive test ──────────────────────────────────────────────────────
    print("\n" + "─" * 95)
    print("Nhập câu lệnh để test thủ công (Enter để thoát):")
    while True:
        try:
            cmd = input("\n> ").strip()
            if not cmd:
                break
            result = predict(cmd, model, tok, lmax)
            print("  Action:")
            for label, val in zip(labels, result["action"]):
                bar = "█" * int(abs(val) / 5) if val != 0 else ""
                sign = "→" if val > 0 else ("←" if val < 0 else "·")
                print(f"    {label:<8} {val:+6.1f}  {sign} {bar}")
        except (KeyboardInterrupt, EOFError):
            break

    print("\nDone.")

if __name__ == "__main__":
    run_tests()
