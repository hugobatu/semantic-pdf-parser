from ContextRecognition import extract_books
import pandas as pd
from pathlib import Path

def export_to_excel(all_reading_lists, filename="course_reading_lists_with_vietnamese_name.xlsx"):
    # default output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    if not filename:
        filename = "course_reading_lists_with_vietnamese_name.xlsx"

    file_path = output_dir / filename

    rows = []
    for course in all_reading_lists:
        raw_list_text = course['reading_list']

        if raw_list_text == "Reading list not found.":
            book_data = []
        else:
            print(f"--- Extracting books for {course['course_id']} ---")
            book_data = extract_books(raw_list_text)

        if not book_data:
            rows.append({
                "Course": course["course_name"],
                "Course ID": course["course_id"],
                "Title": "",
                "Author": "",
                "Year": "",
                "Publisher": ""
            })
        else:
            for b in book_data:
                author = ", ".join(b["author"]) if isinstance(b.get("author"), list) else b.get("author")
                rows.append({
                    "Course": course["course_name"],
                    "Course ID": course["course_id"],
                    "Title": b.get("title"),
                    "Author": author,
                    "Year": b.get("year"),
                    "Publisher": b.get("publisher")
                })

    df = pd.DataFrame(rows)
    df.to_excel(file_path, index=False)
    print(f"✅ Exported {len(rows)} book entries from {len(all_reading_lists)} courses to {file_path}")