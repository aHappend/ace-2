import re

import datasets


def preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def process_doc(doc):
        context = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        return {
            "query": preprocess(doc["activity_label"] + ": " + context),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }

    return dataset.map(process_doc)
