import google.generativeai as genai
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
import os
import re

# for api
load_dotenv() 
api_key = os.getenv("GEMINI_API_KEY")

# check if API key is loaded
if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file")
    print("Current working directory:", os.getcwd())
    print("Files in directory:", os.listdir("."))
    exit(1)

# configure the API key
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# perform pdf ocr if pdf contains no or very few text data extracted as file may contain data in images
def pdf_ocr(pdf_path):
    from pdf2image import convert_from_path
    import pytesseract
    try:
        images = convert_from_path(pdf_path)

        full_text = ""

        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            full_text += text + "\n\n"
        
        print(f"OCR cccompleted successfully.")
        return full_text
    
    except Exception as e:
        print(f"An error occured: {e}")

# main function to extract the content from the pdf
def extract_text_from_pdf(path, use_ocr=False):
    
    if not os.path.exists(path):
        print(f"Error: PDF file '{path}' not found!")
        return ""
    
    # pypdf first (faster for text-based PDFs)
    import pypdf
    if not use_ocr:
        text = ""
        try:
            with open(path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                print(f"PDF has {len(pdf_reader.pages)} pages")
                
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    print(f"Page {i+1} extracted {len(page_text)} characters")
                    if page_text.strip():
                        text += page_text + "\n"
                    else:
                        print(f"Warning: Page {i+1} appears to be empty or image-only")
        
        except Exception as e:
            print(f"Error reading PDF with pypdf: {e}")
            text = ""
        
        print(f"pypdf extraction completed. Total text length: {len(text)} characters")
        
        # if extracted text too less prefer ocr as pdf may contain images
        if len(text.strip()) < 100:
            print("Very little text extracted with pypdf. This might be an image-based PDF.")
            print("Automatically switching to OCR...")
            return pdf_ocr(path)
        else:
            return text
    else:
        return pdf_ocr(path)

# splitting extracted text into smaller chunks
def split_into_chunks(text, max_chunk_size=1000):
    
    if not text or not text.strip():
        print("Warning: No text to split into chunks!")
        return []
    
    sentences = re.split(r'(?<=[.!?]) +', text)
    print(f"Split text into {len(sentences)} sentences")
    
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < max_chunk_size:
            chunk += sentence + " "
        else:
            if chunk.strip(): 
                chunks.append(chunk.strip())
            chunk = sentence + " "
    
    if chunk and chunk.strip():
        chunks.append(chunk.strip())
    
    print(f"Created {len(chunks)} chunks")
    return chunks

# simple embedding of chunks using sentence transformers
def embed_chunks(chunks):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embedder

# gives top k chunks
def get_top_k_chunks(query, embedder, index, chunks, k=3):
    query_vector = embedder.encode([query])
    D, I = index.search(query_vector, k)
    return [chunks[i] for i in I[0]]

# ask gemini ai user queries regarding the pdf content
def ask_gemini(query, context):
    prompt = f"""Based on the following PDF content, answer the user's question.

---PDF Content---
{context}

---Question---
{query}
"""
    return get_response(prompt)

def get_response(message):
    response = model.generate_content(message)
    return response.text if hasattr(response, 'text') else response.candidates[0].text

# --- Main ---
def main():
    """Main function to handle PDF chat with user input"""
    print("=" * 60)
    print("         PDF CHAT ASSISTANT")
    print("=" * 60)
    print("Commands:")
    print("- Type your question to ask about the PDF")
    print("- Type 'load' to load a new PDF file")
    print("- Type 'exit' or 'quit' to stop")
    print("=" * 60)

    chunks = []
    index = None
    embedder = None
    current_pdf = None

    # find and load the user entered pdf file
    def load_pdf():
        nonlocal chunks, index, embedder, current_pdf

        while True:
            pdf_path = input("\nEnter PDF file path (or 'cancel' to go back): ").strip()
            
            if pdf_path.lower() == 'cancel':
                return False
            
            if not pdf_path:
                print("Please enter a valid file path.")
                continue
                
            if not os.path.exists(pdf_path):
                print(f"Error: File '{pdf_path}' not found.")
                print(f"Current directory: {os.getcwd()}")
                print("Make sure the file path is correct.")
                continue
            
            print(f"\n{'='*50}\n")
            print(f"Loading PDF: {pdf_path}")
            print(f"\n{'='*50}")

            try:
                # Check if PDF file exists
                if not os.path.exists(pdf_path):
                    print(f"Error: PDF file '{pdf_path}' not found in current directory")
                    print(f"Current directory: {os.getcwd()}")
                    print(f"Files in directory: {os.listdir('.')}")
                    exit(1)

                text = extract_text_from_pdf(pdf_path, use_ocr=False) 
                
                # check if any content extracted from pdf or not
                if not text or not text.strip():
                    print("ERROR: No text extracted from PDF!")
                    print("This could mean:")
                    print("1. The PDF contains only images/scanned content")
                    print("2. The PDF is password protected")
                    print("3. The PDF has a format that pypdf cannot read")
                    exit(1)
                
                # split extracted text into chunks
                chunks = split_into_chunks(text)
                
                if not chunks:
                    print("ERROR: No chunks created from text!")
                    exit(1)

                # embedd the chunks
                index, embedder = embed_chunks(chunks)
                
                current_pdf = pdf_path
                print(f"\n✅ PDF loaded successfully!")
                print(f"📄 File: {os.path.basename(current_pdf)}")
                print(f"📊 Text length: {len(text):,} characters")
                print(f"📋 Chunks created: {len(chunks)}")
                print("\nYou can now ask questions about this PDF!")
                return True

            except FileNotFoundError:
                print(f"Error: PDF file '{pdf_path}' not found. Please check the file path.")
            except Exception as e:
                print(f"Unexpected error: {e}")
                import traceback
                traceback.print_exc()

    # Initial PDF loading
    if not load_pdf():
        print("No PDF loaded. Exiting...")
        return
    
    # Main chat loop
    print(f"\n{'='*60}")
    print(" CHAT MODE - Ask questions about your PDF!")
    print(f"{'='*60}")
    
    while True:
        try:
            user_input = input(f"\n You (PDF: {os.path.basename(current_pdf) if current_pdf else 'None'}): ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            elif user_input.lower() == 'load':
                if load_pdf():
                    continue
                else:
                    continue
            
            elif user_input.lower() in ['help', 'h']:
                print("\n Available commands:")
                print("- Ask any question about the loaded PDF")
                print("- 'load' - Load a new PDF file")
                print("- 'help' - Show this help message")
                print("- 'exit' or 'quit' - Exit the program")
                continue
            
            # Check if PDF is loaded
            if not chunks or not index or not embedder:
                print(" No PDF loaded! Use 'load' command to load a PDF first.")
                continue
            
            # Process the question
            print("🔍 Searching relevant content...")
            relevant_chunks = get_top_k_chunks(user_input, embedder, index, chunks, k=5)
            context = "\n".join(relevant_chunks)
            
            if not context:
                print("No relevant content found for your question.")
                continue
            
            print("Thinking...")
            answer = ask_gemini(user_input, context)
            
            print(f"\n Assistant: {answer}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\Error: {e}")

# entry point
if __name__ == "__main__":
    main()