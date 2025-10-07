import spacy
from pathlib import Path

current_dir = Path(__file__).parent
model_path = current_dir.parent / "arxiv_ner_model"
model = spacy.load(model_path)