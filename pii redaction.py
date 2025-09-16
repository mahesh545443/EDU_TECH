import os
import pypdf
import json
import re
import base64
import fitz
import docx
import openpyxl
from io import BytesIO
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.exceptions import OutputParserException

# --- Pydantic Schemas for Structured Output ---
class RedactionResult(BaseModel):
    """The result of a redaction task, containing the redacted text and a list of identified PII."""
    redacted_text: str = Field(description="The full document with all PII replaced by placeholders.")
    detected_pii: List[str] = Field(description="A list of all detected PII entities for logging purposes.")

# --- LLM and Agentic Pipeline Configuration ---
API_KEY = "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"

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=API_KEY,
    max_tokens=500,  # ✅ prevent exceeding free-tier credit limit
    default_headers={  # ✅ OpenRouter requires proper headers
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "http://localhost",  # you can replace with your app/site
        "X-Title": "PII Redaction Script"
    }
)

redaction_llm = llm.with_structured_output(RedactionResult)

# --- File Handling and Processing ---

def process_file_to_message(file_path: str) -> Optional[HumanMessage]:
    """
    Reads a document and returns a HumanMessage formatted for a multimodal LLM.
    This handles text-based files and images without Tesseract.
    """
    try:
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext in ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.tiff']:
            print(f" - Processing {os.path.basename(file_path)} as a multimodal file...")
            if file_ext == '.pdf':
                DPI = 300
                doc = fitz.open(file_path)
                page = doc.load_page(0)
                pix = page.get_pixmap(dpi=DPI)
                img_bytes = pix.tobytes("png")
                doc.close()
                base64_image = base64.b64encode(img_bytes).decode("utf-8")
            else:
                with open(file_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
            strong_prompt = """
            You are a highly specialized and precise PII redaction agent. Your task is to analyze the image of a document and return the full text with all PII replaced by specific, descriptive placeholders.

            Your goal is to preserve the original document's structure and formatting as much as possible while completely redacting sensitive information.

            Instructions for Redaction:
            1.  **Replace PII** with the most appropriate placeholder from this list: [NAME], [DATE], [ID_NUMBER], [ADDRESS], [EMAIL], [PHONE_NUMBER], [FINANCIAL_INFO].
            2.  **Be Precise**: Only redact the PII itself. Do not redact surrounding text, punctuation, or formatting.
            3.  **Preserve Layout**: Maintain the original line breaks, spacing, and overall structure of the document.
            4.  **Handle IDs**: For document IDs like a PAN or Aadhar, replace the entire number with the placeholder [ID_NUMBER]. For example, `CVDPM6627B` becomes `[ID_NUMBER]`.
            5.  **For Dates**: For dates of birth or issue dates, use the placeholder `[DATE]`.

            Return the full redacted text. Also, provide a list of the PII entities you detected for logging.
            """
            
            return HumanMessage(content=[
                {"type": "text", "text": strong_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            ])

        elif file_ext in ['.txt', '.docx', '.xlsx', '.xls']:
            print(f" - Processing {os.path.basename(file_path)} as a text-based file...")
            text_content = ""
            if file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            elif file_ext == '.docx':
                doc = docx.Document(file_path)
                text_content = "\n".join([para.text for para in doc.paragraphs])
            elif file_ext in ['.xlsx', '.xls']:
                wb = openpyxl.load_workbook(file_path, data_only=True)
                for sheet in wb.worksheets:
                    for row in sheet.iter_rows():
                        text_content += " ".join([str(cell.value) if cell.value is not None else "" for cell in row]) + "\n"

            strong_prompt = f"""
            You are a highly specialized and precise PII redaction agent. Your task is to analyze the following document text and return the full text with all PII replaced by specific, descriptive placeholders.

            Your goal is to preserve the original document's structure and formatting as much as possible while completely redacting sensitive information.

            Instructions for Redaction:
            1.  **Replace PII** with the most appropriate placeholder from this list: [NAME], [DATE], [ID_NUMBER], [ADDRESS], [EMAIL], [PHONE_NUMBER], [FINANCIAL_INFO].
            2.  **Be Precise**: Only redact the PII itself. Do not redact surrounding text, punctuation, or formatting.
            3.  **Preserve Layout**: Maintain the original line breaks, spacing, and overall structure of the document.
            4.  **Handle IDs**: For document IDs like a PAN or Aadhar, replace the entire number with the placeholder [ID_NUMBER].
            5.  **For Dates**: For dates of birth or issue dates, use the placeholder `[DATE]`.

            Return the full redacted text. Also, provide a list of the PII entities you detected for logging.

            Document text:
            ---
            {text_content}
            ---
            """
            return HumanMessage(content=strong_prompt)

        else:
            print(f" - File type {file_ext} is not supported. Skipping.")
            return None

    except Exception as e:
        print(f" - Error processing file {file_path}: {e}")
        return None

# --- Main Automation Loop ---
def process_and_report_folder(folder_path: str):
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at '{folder_path}'.")
        return

    print(f"Starting automated PII redaction for folder: {folder_path}\n")
    
    file_count = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            file_count += 1
            
            message = process_file_to_message(file_path)
            if not message:
                continue

            try:
                print(f" - Sending to LLM for redaction...")
                result = redaction_llm.invoke([message])
                
                print("\n" + "="*80)
                print(f"✨ Redaction Report for: {filename} ✨".center(80))
                print("="*80 + "\n")

                print("### Detected PII")
                print("-" * 80)
                if result.detected_pii:
                    for i, pii in enumerate(result.detected_pii, 1):
                        print(f" {i}. {pii}")
                else:
                    print("   No PII was detected.")
                print("-" * 80)
                
                print("\n### Redaction Result")
                print("-" * 80)
                print(result.redacted_text)
                print("-" * 80)

                print("\n" + "="*80 + "\n")
            
            except Exception as e:
                print(f" - An error occurred during processing: {e}")

    if file_count == 0:
        print("No supported documents found in the folder.")
    else:
        print(f"Finished processing {file_count} documents.")

# --- Main Execution Block ---
if __name__ == "__main__":
    os.environ['OPENROUTER_API_KEY'] = API_KEY
    documents_folder_path = r"C:\Users\User\ocr_samples"
    process_and_report_folder(documents_folder_path)
