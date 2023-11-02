from datasets import load_dataset
import os
import tempfile
from git import Repo


def get_conll():
    """
    Get the CoNLL-2003 dataset.
    """
    print("Getting CoNLL-2003")
    dataset = load_dataset("conll2003")
    id2label = dict(enumerate(dataset["train"].features["ner_tags"].feature.names))

    for split in ["train", "validation", "test"]:
        with open(
            f"data/en.conll.{split}.tsv",
            "w",
            encoding="utf8",
        ) as f:
            for example in dataset[split]:
                tokens = example["tokens"]
                labels = example["ner_tags"]
                labels = [id2label[label] for label in labels]
                for token, label in zip(tokens, labels):
                    f.write(f"{token}\t{label}\n")
                f.write("\n")

        # Remove MISC labels
        with open(
            f"data/en.conll.{split}.nomisc.tsv",
            "w",
            encoding="utf8",
        ) as f:
            for example in dataset[split]:
                tokens = example["tokens"]
                labels = example["ner_tags"]
                # Remove MISC labels
                labels = [id2label[label] for label in labels]
                for i in range(len(labels)):
                    if "MISC" in labels[i]:
                        labels[i] = "O"
                for token, label in zip(tokens, labels):
                    f.write(f"{token}\t{label}\n")
                f.write("\n")


def get_masakhaner2():
    """
    Get the Masakhaner2 dataset.
    """
    print("Getting Masakhaner2")
    for lang in ["hau", "ibo", "nya", "sna", "swa", "xho", "yor", "zul"]:
        dataset = load_dataset("masakhane/masakhaner2", lang)
        id2label = dict(enumerate(dataset["train"].features["ner_tags"].feature.names))

        for split in ["train", "validation", "test"]:
            with open(
                f"data/{lang}.masakhaner2.{split}.tsv",
                "w",
                encoding="utf8",
            ) as f:
                for example in dataset[split]:
                    tokens = example["tokens"]
                    labels = example["ner_tags"]
                    labels = [id2label[label] for label in labels]
                    # We only want LOC, PER and ORG
                    for i in range(len(labels)):
                        if (
                            "LOC" not in labels[i]
                            and "PER" not in labels[i]
                            and "ORG" not in labels[i]
                        ):
                            labels[i] = "O"

                    for token, label in zip(tokens, labels):
                        f.write(f"{token}\t{label}\n")
                    f.write("\n")


def get_europarl():
    print("Getting Europarl-Ner Corpus")
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Download the following GitHub repo: https://github.com/ixa-ehu/ner-evaluation-corpus-europarl
        Repo.clone_from(
            "https://github.com/ixa-ehu/ner-evaluation-corpus-europarl", tmpdirname
        )
        os.rename(f"{tmpdirname}/en-europarl.test.conll02", "data/en.europarl.test.tsv")
        os.rename(f"{tmpdirname}/es-europarl.test.conll02", "data/es.europarl.test.tsv")
        os.rename(f"{tmpdirname}/de-europarl.test.conll02", "data/de.europarl.test.tsv")
        os.rename(f"{tmpdirname}/it-europarl.test.conll02", "data/it.europarl.test.tsv")


def get_ote():
    label2id = {"O": 0, "B-TARGET": 1, "I-TARGET": 2}
    id2label = {v: k for k, v in label2id.items()}
    print("Getting OTE")
    for lang in ["en", "es", "fr", "ru", "tr"]:
        dataset = load_dataset("HiTZ/Multilingual-Opinion-Target-Extraction", lang)

        for split in ["train", "test"] if lang != "tr" else ["train"]:
            with open(
                f"data/{lang}.ote.{split}.tsv",
                "w",
                encoding="utf8",
            ) as f:
                for example in dataset[split]:
                    tokens = example["tokens"]
                    labels = example["ner_tags"]
                    labels = [id2label[label] for label in labels]
                    # We only want LOC, PER and ORG
                    for i in range(len(labels)):
                        if (
                            "LOC" not in labels[i]
                            and "PER" not in labels[i]
                            and "ORG" not in labels[i]
                        ):
                            labels[i] = "O"

                    for token, label in zip(tokens, labels):
                        f.write(f"{token}\t{label}\n")
                    f.write("\n")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    get_conll()
    get_masakhaner2()
    get_europarl()
    get_ote()


