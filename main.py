from text_extraction import record_text_extraction, get_pdf_paths


if __name__ == "__main__":
    pdf_paths = get_pdf_paths("raw-pdfs")

    test_doc_path = pdf_paths[3]
    print(test_doc_path)

    print(record_text_extraction(test_doc_path))
