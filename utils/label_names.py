from typing import Dict

_label2name: Dict[str, str] = {
    "LOC": "Location",
    "PER": "Person",
    "ORG": "Organization",
    "MISC": "Miscellaneous",
    "TARGET": "Target",
    "ety": "ClinicalEntity",
    "dis": "Disability",
}

_name2label: Dict[str, str] = {v: k for k, v in _label2name.items()}


def label2name(label: str) -> str:
    if label in _label2name:
        return _label2name[label]
    else:
        return label


def name2label(name: str) -> str:
    if name in _name2label:
        return _name2label[name]
    else:
        return name
