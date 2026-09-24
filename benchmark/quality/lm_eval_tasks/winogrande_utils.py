def doc_to_text(doc):
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[doc["answer"]]


def doc_to_target(doc):
    index = doc["sentence"].index("_") + 1
    return doc["sentence"][index:].strip()


def doc_to_choice(doc):
    index = doc["sentence"].index("_")
    return [
        doc["sentence"][:index] + option
        for option in (doc["option1"], doc["option2"])
    ]
