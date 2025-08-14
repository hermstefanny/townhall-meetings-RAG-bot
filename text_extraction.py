import fitz
import os


def get_pdf_paths(dir_name) -> list[str]:
    script_directory = os.getcwd()
    path_to_folder = os.path.join(script_directory, dir_name)

    pdf_paths = [
        os.path.abspath(os.path.join(path_to_folder, f))
        for f in os.listdir(path_to_folder)
        if f.lower().endswith(".pdf")
    ]

    return pdf_paths


def record_text_extraction(record_file_path) -> str:

    try:
        record_doc = fitz.open(record_file_path)
    except Exception:
        print("File path not found")
        return ""

    # Hard coded header and footer values according to layout document
    header_space = 0.165
    footer_space = 0.925

    record_text = ""

    for num_page in range(len(record_doc)):
        page = record_doc.load_page(num_page)

        page_height = page.rect.height
        page_width = page.rect.width

        text_rect = fitz.Rect(
            0, page_height * header_space, page_width, page_height * footer_space
        )

        page_text = page.get_text("text", clip=text_rect)

        record_text += page_text

    record_doc.close()

    return record_text
