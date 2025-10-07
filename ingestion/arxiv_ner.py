from collections import defaultdict
from ner_model import model

def group_by_entity(doc_details):
    for doc in doc_details:
        ner_doc = model(doc['page_content'])
        pairs = [(ent.text, ent.label_) for ent in ner_doc.ents]
        entity_dict = defaultdict(list)
        for word, entity in pairs:
            entity_dict[entity].append(word)
        temp_dict = dict(entity_dict)
        doc.update(temp_dict)
    return doc_details