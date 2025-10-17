# 🧠 Semantic PDF Parser

A powerful and modular pipeline designed to **extract**, **understand**, and **structure semantic information** from PDF documents — integrating OCR, layout detection, and intelligent content parsing for high-accuracy text analysis based on Rednote Hilab Dots.Ocr.

---

## Architecture Overview (This could have been done if I have a better GPU for running OCR model)

<img src="assets\PDF_Semantic Parser_Architecture.drawio.png" alt="Architecture Diagram"/>

The system follows a microservice-based architecture, separating core functionalities into independent services such as OCR, layout analysis, text classification, and semantic structuring.

---

## 🚀 Features

- 🔤 **OCR Integration:** Uses advanced 1.7B-parameter model OCR for text recognition, detects paragraphs, tables, and other document structures..
- 🧠 **Semantic Understanding:** Extracts key entities, context, and relationships.
- 🔍 **Flexible Output:** Structured JSON for Gemini API processing with further unclear information of Reading list data.
- 🐳 **Dockerized Services**

---

## 🧱 Project Structure
Again, wish I could have a better GPU for running OCR model, so this project is currently at the idea-level 😭