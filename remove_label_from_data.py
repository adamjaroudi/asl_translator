"""
Remove one or more labels (letters or words) from the saved training data.

Use this when you already saved bad data (e.g. wrong "E") and want to delete
those rows from data/landmarks.csv.

Usage:
    python remove_label_from_data.py E           # remove all rows with label E
    python remove_label_from_data.py E F         # remove E and F
    python remove_label_from_data.py [HELLO]      # remove word sign [HELLO]
"""

import os
import csv
import sys
import tempfile
import shutil

DATA_FILE = os.path.join("data", "landmarks.csv")


def main():
    if len(sys.argv) < 2:
        print("Usage: python remove_label_from_data.py <label> [label2 ...]")
        print("Example: python remove_label_from_data.py E    (removes all 'E' rows)")
        print("         python remove_label_from_data.py E F  (removes E and F)")
        return 1

    labels_to_remove = [arg.strip().upper() if not arg.startswith("[") else arg.strip() for arg in sys.argv[1:]]

    if not os.path.exists(DATA_FILE):
        print(f"File not found: {DATA_FILE}")
        return 1

    with open(DATA_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("File is empty.")
        return 0

    header = rows[0]
    data_rows = rows[1:]
    label_col = 0

    removed = {label: 0 for label in labels_to_remove}
    new_rows = []
    for row in data_rows:
        if len(row) <= label_col:
            continue
        lab = row[label_col].strip()
        key = lab.upper() if len(lab) == 1 else lab
        if key in removed:
            removed[key] += 1
            continue
        new_rows.append(row)

    total_removed = sum(removed.values())
    try:
        # Write to temp file first, then copy over (works on all Python versions)
        fd, tmp_path = tempfile.mkstemp(suffix=".csv", dir=os.path.dirname(DATA_FILE) or ".")
        try:
            with os.fdopen(fd, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(new_rows)
            shutil.copy2(tmp_path, DATA_FILE)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except PermissionError:
        print(f"Permission denied: could not write to {DATA_FILE}")
        print("  Close the file if it's open in Excel, another editor, or Cursor, then try again.")
        return 1

    print(f"Removed {total_removed} row(s) from {DATA_FILE}")
    for label in labels_to_remove:
        n = removed.get(label, 0)
        if n:
            print(f"  {label}: {n} samples removed")
    print(f"Remaining: {len(new_rows)} samples. Retrain with: python train_model.py --accuracy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
