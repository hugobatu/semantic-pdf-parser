# 🧠 Semantic PDF Parser

## 1. Overview
This project automates the extraction and structuring of course reading lists from academic module handbooks (PDFs). It uses an LLM for context recognition and maps English course names to their Vietnamese equivalents, producing a final structured Excel file for downstream academic or administrative use.

---

## 2. 🏗️ Architecture Overview

<img src="assets\PDF_Semantic Parser_Architecture.drawio_new.png" alt="Architecture Diagram"/>

### 2.1. Input: Physics Courses’ Module Handbook (Full English)
Multiple PDF files containing full module handbooks for physics courses serve as input.
### 2.2. PDF Parser
Parses the PDF documents to extract textual content, focusing on reading lists per course. Outputs a structured dataset containing:
- Course name and ID
- Raw reading list text

### 2.3. LLM Processing: Mistral 7B Instruct v0.2
Utilizes the Mistral 7B Instruct v0.2 model via OpenRouter to interpret the extracted reading lists and generate:

- Structured metadata for each book
    - Title
    - Author(s)
    - Year
    - Publisher
- Linked course information

This step is handled by the ContextRecognition module, which uses the model to extract semantic meaning from raw text.

### 2.4. Mapping
A mapping step merges the recognized course data with a separate Excel file containing:
- Vietnamese course names
- Corresponding course IDs

→ This produces a unified dataset aligning English and Vietnamese course information.

### 2.5. Output
Exports the final dataset to Excel including:
- Vietnamese Course Name
- Course ID
- Book Title
- Author
- Year
- Publisher

Default output location: ***`./exports/course_reading_lists.xlsx`***
## 3. 📊 Monitoring
The system includes monitoring via Prometheus and Grafana, enabling:
- Resource tracking (CPU, memory, VRAM)
- Visualization dashboards for performance insights

## 🐳 Docker Integration
The entire pipeline is containerized for easy deployment.
Docker handles environment setup, dependencies, and service orchestration including services:
- PDF Parser service
- Context Recognition service (LLM-based)
- Monitoring stack (Prometheus + Grafana)

## 🚀 Features
1. Drop your module handbook PDFs into the input folder.
2. Run the parser container to extract structured text.
3. The ContextRecognition module setting up the Mistral model and then running locally to extract book metadata.
4. The resulting records are merged with the Vietnamese course mapping file.
5. The final Excel report is automatically generated in /exports/

---

## 🧱 Project Structure

**Detailed breakdown:**

*   **`src/`**: Contains all the application's source code.
    *   **`ContextRecognition.py`**: LLM model for book context recognition setting up and running locally.
    *   **`ExcelExporter.py`**: Exporting final data to excel file format.
    *   **`Mapper.py`**: Mapping the Vietnamese courses' name using course_id with the English version.
    *   **`TableOfContentExtractor.py`**: Extracting the table of contents for getting the courses's name in English
    *   **`requirements.txt`**: environemnt dependencies for running the project