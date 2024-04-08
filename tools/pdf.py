import os
import uuid
from pathlib import Path

import pdfkit
from langchain.tools import tool
from markdown import markdown
from pydantic import BaseModel, Field


PATH_WKHTMLTOPDF = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
PDFKIT_CONFIG = pdfkit.configuration(wkhtmltopdf=PATH_WKHTMLTOPDF)

OUTPUT_DIRECTORY = Path(__file__).parent.parent / "output"


class MarkdownToPDFInput(BaseModel):
    markdown_text: str = Field(
        description="Markdown text to convert to PDF, provided in valid markdown format."
    )


def generate_html_text(markdown_text: str) -> str:
    """Convert markdown text to HTML text."""
    markdown_text = markdown_text.replace("file:///", "").replace("file://", "")
    html_text = markdown(markdown_text)
    html_text = f"""
    <html>
    <head>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto&display=swap');
            body {{
                font-family: 'Roboto', sans-serif;
                line-height: 150%;
            }}
        </style>
    </head>
    <body>
    {html_text}
    </body>
    </html>
    """
    return html_text


@tool("markdown_to_pdf_file", args_schema=MarkdownToPDFInput)
def markdown_to_pdf_file(markdown_text: str) -> str:
    """Convert markdown text to a PDF file. Takes valid markdown as a string as input and will return a string file-path to the generated PDF."""
    html_text = generate_html_text(markdown_text)
    unique_id: uuid.UUID = uuid.uuid4()
    pdf_path = OUTPUT_DIRECTORY / f"{unique_id}.pdf"

    options = {
        "no-stop-slow-scripts": True,
        "print-media-type": True,
        "encoding": "UTF-8",
        "enable-local-file-access": "",
    }

    pdfkit.from_string(
        html_text, str(pdf_path), configuration=PDFKIT_CONFIG, options=options
    )

    if os.path.exists(pdf_path):
        return str(pdf_path)
    else:
        return "Could not generate PDF, please check your input and try again."


markdown_dummy_text = """
# Title
This is a test of the markdown to PDF function.
## Subtitle
This is a test of the markdown to PDF function.
### Sub-subtitle
This is a test of the markdown to PDF function. This is a paragraph with random text in it nunc nunc tincidunt nunc, nec.
S'il vous plaît.
"""

if __name__ == "__main__":
    print(markdown_to_pdf_file(markdown_dummy_text))
