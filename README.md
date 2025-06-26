# PDF Analyst CLI

A command-line application for analyzing PDF documents and answering user questions based on their content.

## Features

* Extracts text from PDF documents.
* Converts PDF pages to images for OCR.
* Processes document content to answer natural language queries.

## Installation

1.  Clone this repository:
    `git clone <repository_url>`
    `cd pdf-analyst-cli`

2.  Create and activate a virtual environment:
    `python -m venv .venv`
    `.\.venv\Scripts\activate` (Windows)

3.  Install dependencies:
    `pip install -r requirements.txt`

    *(Ensure you have Tesseract OCR and Poppler installed on your system for full functionality.)*

## Usage

Run the main script with your PDF file:

`python main.py <path_to_your_pdf.pdf>`

Follow the prompts to ask questions about the PDF content.
