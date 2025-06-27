# Document Chat Assistant
A command-line application that analyzes documents and answers user questions based on their content.

## Features
* Extracts text from Document Files.
* Processes document content to answer natural language queries.
* Supported file formats- pdf, docx, csv, txt, xlsx, pptx

## Installation
* Clone this repository:
`git clone <repository_url>`
`cd document-chat-assistant`

* Create and activate a virtual environment:
`python -m venv .venv` <br>
  On Windows:
`.venv\Scripts\activate` <br>
  On macOS/Linux:
`source .venv/bin/activate`

* Install dependencies:
`pip install -r requirements.txt`

(Note: This application requires Tesseract OCR and Poppler to be installed on your system for full functionality.)

## Usage
Run the main script:
`python main.py`

The application will then prompt you to enter the path to your file.
