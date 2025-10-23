import pdfplumber
import re
from typing import List, Dict, Any

# PDF parser for physics courses handbook
def extract_course_reading_lists(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts course details and the raw text of their reading lists from the PDF.
    """
    course_details: List[Dict[str, Any]] = []

    with pdfplumber.open(pdf_path) as pdf:
        # Step 1: find end of TOC
        # (Your original logic for finding the TOC)
        toc_end_page_index = 0
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            if tables:
                toc_end_page_index = page_num - 1
                break

        # Step 2: parse TOC
        # (Your original logic for parsing the TOC)
        full_toc_text = "\n".join(
            page.extract_text() for page in pdf.pages[:toc_end_page_index] if page.extract_text()
        )
        pattern = re.compile(
            r"^\s*(\d+)\.\s+(.*?)\s*[–—-]\s*([A-Z]{3,}\d+)[\s\.]+(\d+)",
            re.MULTILINE
        )
        raw_entries = pattern.findall(full_toc_text)
        toc_entries = []
        for num, name, code, page_num_str in raw_entries:
            cleaned_name = re.sub(r'\s*\.{2,}.*$', '', name.replace('\n', ' ')).strip()
            toc_entries.append({"num": num, "name": cleaned_name, "id": code, "page": int(page_num_str)})
        
        print(f"Found {len(toc_entries)} TOC entries.")

        # Step 3: combine all course detail pages
        # *** MODIFIED: Use page.crop() to remove headers/footers (page numbers) ***
        all_page_texts = []
        for page in pdf.pages[toc_end_page_index:]:
            # Crop top 5% and bottom 5%
            h = float(page.height)
            w = float(page.width)
            cropped_page = page.crop((0, h * 0.05, w, h * 0.95))
            
            text = cropped_page.extract_text()
            if text:
                all_page_texts.append(text)
        
        full_text = "\n".join(all_page_texts)

        # Step 4: find *all* course markers first
        markers = []
        for course in toc_entries:
            pattern = (
                re.escape(course['num']) +
                r"\.\s+" +
                re.escape(course['name']) +
                r"\s*[–—-]\s*" +  # allow any dash
                re.escape(course['id'])
            )
            match = re.search(pattern, full_text, flags=re.IGNORECASE)
            if match:
                markers.append((match.start(), course))
            else:
                print(f"❌ Could not find {course['num']} {course['id']}")

        markers.sort(key=lambda x: x[0])

        # Step 5: slice sections properly
        for i, (start_idx, course) in enumerate(markers):
            end_idx = len(full_text)
            if i + 1 < len(markers):
                end_idx = markers[i+1][0]

            course_text = full_text[start_idx:end_idx]

            # *** MODIFIED: Simpler reading list text extraction ***
            reading_list_marker = "Reading list"
            reading_list_text = "Reading list not found." # Default
            
            if reading_list_marker in course_text:
                try:
                    # Find the start of the text *after* the marker
                    start_list_idx = course_text.index(reading_list_marker) + len(reading_list_marker)
                    raw_list_text = course_text[start_list_idx:]
                    
                    # Clean up: remove leading newlines and collapse multiple empty lines
                    raw_list_text = re.sub(r'^\s*[\n\r]+', '', raw_list_text)
                    raw_list_text = re.sub(r'\n\s*\n+', '\n', raw_list_text).strip()
                    
                    if raw_list_text: # Ensure we got something
                         reading_list_text = raw_list_text
                except Exception as e:
                    print(f"Error parsing reading list for {course['id']}: {e}")

            course_details.append({
                "course_name": course['name'],
                "course_id": course['id'],
                "reading_list": reading_list_text # This now holds the raw string
            })

    return course_details