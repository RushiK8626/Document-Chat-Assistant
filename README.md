# PDF Analyst CLI
A command-line application that analyzes PDF documents and answers user questions based on their content.

## Features
* Extracts text from PDF documents.
* Processes document content to answer natural language queries.

## Installation
* Clone this repository:
`git clone <repository_url>`
`cd pdf-analyst-cli`

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

The application will then prompt you to enter the path to your PDF file.
