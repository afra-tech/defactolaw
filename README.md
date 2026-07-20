# DefactoLaw

An open-source AI-powered legal contract review application. Uses a fine-tuned RoBERTa model (CUAD dataset) to automatically extract and highlight key contractual clauses from uploaded PDF contracts.

[![Paper](https://img.shields.io/badge/Paper-ACL%202022-blue)](https://aclanthology.org/2022.lateraisse-1.6/)
[![Model](https://img.shields.io/badge/Model-Hugging%20Face-yellow)](https://huggingface.co/Rakib/roberta-base-on-cuad)
[![License](https://img.shields.io/badge/License-AGPL%20v3-green)](LICENSE)

## Features

- **PDF Upload** – Drag-and-drop contract upload via web interface
- **OCR Support** – Scanned PDFs are automatically OCR'd via Tesseract before analysis
- **AI Clause Extraction** – 41 predefined legal question types (indemnification, confidentiality, governing law, etc.) answered by a RoBERTa QA model fine-tuned on CUAD
- **Color-Coded Highlighting** – Predicted clauses are highlighted directly in the PDF, each clause type with its own color
- **Interactive Viewer** – Sidebar navigation to inspect clause predictions incrementally

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Django 3.2, Python 3.9 |
| ML | Hugging Face Transformers, PyTorch (CPU), RoBERTa-base on CUAD |
| PDF Processing | PyMuPDF, pdfminer.six, pdf2image |
| OCR | Tesseract (via pytesseract), OpenCV |
| Frontend | Django Templates, Bootstrap 5.1, vanilla JS |
| Database | SQLite (dev) / PostgreSQL (prod via `DATABASE_URL`) |
| Deployment | Gunicorn, Whitenoise (Heroku-ready) |

## Installation

### Prerequisites

- Python 3.9.6
- Tesseract OCR engine
- Poppler (for pdf2image)

### Setup

```bash
git clone https://github.com/sanaullahaq/defactolaw.git
cd defactolaw
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

## Usage

1. Open `http://localhost:8000/`
2. Upload a contract PDF
3. Click on any clause type in the sidebar to run inference
4. Predicted text spans are highlighted in the PDF with color-coded annotations

## ML Model

The model is a fine-tuned `roberta-base` trained on the [CUAD dataset](https://github.com/TheAtticusProject/cuad). It is hosted on Hugging Face Hub as [`Rakib/roberta-base-on-cuad`](https://huggingface.co/Rakib/roberta-base-on-cuad). A local copy is managed via Git LFS.

## Project Structure

```
defactolaw/
├── contractweb/          # Django project settings/urls
├── contracts/            # Main app (views, models, forms, OCR logic)
├── templates/            # HTML templates, CSS, JS, Bootstrap
├── files/                # Uploaded PDFs
├── tessdata/             # Tesseract language data
├── E4_roberta-base/      # ML model weights (Git LFS)
├── requirements.txt
└── runtime.txt           # Python version (Heroku)
```

## Citation

```bibtex
@inproceedings{nawar-etal-2022-open,
    title = "An Open Source Contractual Language Understanding Application Using Machine Learning",
    author = "Nawar, Afra  and
      Rakib, Mohammed  and
      Hai, Salma Abdul  and
      Haq, Sanaulla",
    editor = "Adebayo, Kolawole  and
      Nanda, Rohan  and
      Verma, Kanishk  and
      Davis, Brian",
    booktitle = "Proceedings of the First Workshop on Language Technology and Resources for a Fair, Inclusive, and Safe Society within the 13th Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lateraisse-1.6/",
    pages = "42--50"
}
```

## License

This project is licensed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE).
