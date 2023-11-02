from torch.utils.data import Dataset, DataLoader
import os
import json
from utils.utils import (
    count_predictions,
    get_labels,
    read_all_sentences_tsv,
    format_sentence_with_labels,
    get_task_labels,
    check_labels_have_punctuation,
)
from typing import List, Dict, Union, Tuple
from transformers import PreTrainedTokenizer, DataCollatorForSeq2Seq
from tqdm.auto import tqdm
import torch
from functools import partial


def prepare_pair(
    tokenizer: PreTrainedTokenizer,
    source_lang: str,
    target_lang: str,
    source_txt: str,
    target_txt: str,
    max_length: int,
    decoder_start_token_id: int,
):
    tokenizer.src_lang = source_lang
    tokenizer.tgt_lang = target_lang
    inputs = tokenizer(
        source_txt,
        return_tensors=None,
        padding=False,
        truncation=True,
        max_length=max_length,
    )

    labels = tokenizer(
        target_txt=target_txt,
        return_tensors=None,
        padding=False,
        truncation=True,
        max_length=max_length,
    )

    # labels["input_ids"].insert(0, decoder_start_token_id)
    labels["input_ids"].append(tokenizer.eos_token_id)
    # labels["input_ids"][labels["input_ids"] == tokenizer.pad_token_id] = -100

    inputs["decoder_input_ids"] = labels["input_ids"]

    return inputs


def prepare_sl(
    tokenizer,
    x: str,
    y: str,
    source_words: str,
    source_entities: str,
    source_labels: str,
    target_words: str,
    target_entities: str,
    target_labels: str,
    task_name: str,
    idx: int,
    max_source_len: int,
    max_target_len: int,
) -> List[Union[Dict[str, Union[torch.tensor, str]], List[str]]]:
    model_inputs = tokenizer(
        x,
        max_length=max_source_len,
        padding=False,
        truncation=True,
        return_tensors=None,
    )


    y_tokenized = tokenizer(
        text_target=y,
        max_length=max_target_len,
        padding=False,
        truncation=True,
        return_tensors=None,
    )
    model_inputs["labels"] = y_tokenized["input_ids"]

    task_name = tokenizer(
        task_name,
        max_length=1024,
        padding=False,
        truncation=True,
        return_tensors=None,
    )

    source_words = tokenizer(
        source_words,
        max_length=1024,
        padding=False,
        truncation=True,
        return_tensors=None,
    )

    source_entities = tokenizer(
        source_entities,
        max_length=1024,
        padding=False,
        truncation=True,
        return_tensors=None,
    )

    source_labels = tokenizer(
        source_labels,
        max_length=1024,
        padding=False,
        truncation=True,
        return_tensors=None,
    )

    target_words = tokenizer(
        target_words,
        max_length=1024,
        padding=False,
        truncation=True,
        return_tensors=None,
    )

    target_entities = tokenizer(
        target_entities,
        max_length=1024,
        padding=False,
        truncation=True,
        return_tensors=None,
    )

    target_labels = tokenizer(
        target_labels,
        max_length=1024,
        padding=False,
        truncation=True,
        return_tensors=None,
    )

    return [
        model_inputs,
        task_name,
        idx,
        source_words,
        source_entities,
        source_labels,
        target_words,
        target_entities,
        target_labels,
    ]


class ScoreDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        source_lang: str,
        target_lang: str,
        decoder_start_token_id: int,  # model.config.decoder_start_token_id
        normalize: bool = True,
        both_directions: bool = True,
        max_length: int = 512,
        top_k: int = None,
    ):
        """
        {"New York": ["Nueva York","York","Nueva"]}
        We built a dictionary to ensure that we don't calculate the probability of the pair twice.
        Building a dictionary is very fast, running m2m100_12B is not.
        """

        print(
            f"Loading {jsonl_path}.\n"
            f"source_lang={source_lang}, target_lang={target_lang}.\n"
            f"max_length={max_length}, top_k={top_k}, normalize={normalize}, both_directions={both_directions}.\n"
        )

        task_labels = get_task_labels(jsonl_path)
        labels_have_punctuation = check_labels_have_punctuation(jsonl_path)

        data_dictionary: Dict[str, List[str]] = {}
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                json_dict: Dict[str, Union[str, List[str]]] = json.loads(line)
                source_labels: str = json_dict["source_entity"].strip()
                predictions: List[str] = json_dict["preds"]
                target_sentence: str = json_dict["target_word"].strip()
                target_words: List[str] = target_sentence.split()

                prediction_counter: Dict[str, Dict[str, int]] = count_predictions(
                    predictions=predictions,
                    target_words=target_words,
                    top_k=top_k,
                    task_labels=task_labels,
                    labels_have_punctuation=labels_have_punctuation,
                )

                source_labels, source_label_types = get_labels(
                    source_labels,
                    possible_labels=task_labels,
                    labels_have_punctuation=labels_have_punctuation,
                )
                for text, label_class in zip(source_labels, source_label_types):
                    if text not in data_dictionary:
                        data_dictionary[text] = []
                    if label_class in prediction_counter:
                        data_dictionary[text].extend(
                            list(prediction_counter[label_class].keys())
                        )

        pairs: List[Tuple[str, str, str, str]] = []
        self.data = []

        for source, targets in data_dictionary.items():
            for target in targets:
                # pairs.append((target, source, target_lang, source_lang))
                pairs.append((source, target, source_lang, target_lang))

                if both_directions:
                    # pairs.append((source, target, source_lang, target_lang))
                    pairs.append((target, source, target_lang, source_lang))

                    if normalize:
                        pairs.append((target, target, target_lang, target_lang))

            if normalize:
                pairs.append((source, source, source_lang, source_lang))

        # Convert to set to remove duplicates
        for source, target, source_lang, target_lang in tqdm(
            set(pairs), desc="Tokenizing dataset"
        ):
            self.data.append(
                prepare_pair(
                    tokenizer=tokenizer,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    source_txt=source,
                    target_txt=target,
                    max_length=max_length,
                    decoder_start_token_id=decoder_start_token_id,
                )
            )

        print(f"Number of pairs to evaluate: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class T5Dataset(Dataset):
    def __init__(
        self,
        source_tsv_path: str,
        target_tsv_path: str,
        tokenizer,
        max_source_len: int = 128,
        max_target_len: int = 128,
        inference: bool = False,
    ):
        print(f"Loading source sentences from {source_tsv_path}")

        self.source_tsv_path = source_tsv_path
        self.target_tsv_path = target_tsv_path
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

        self.target_sentences = {}

        # LOAD DATA

        source_dataset_sentences = read_all_sentences_tsv(
            self.source_tsv_path,
        )
        source_dataset_sentences = list(map(list, zip(*source_dataset_sentences)))

        if self.target_tsv_path.endswith(".txt"):
            target_dataset_sentences = []
            print(
                "Your target file is a txt file, we will convert it to a tsv file with O labels for all the words"
            )
            lines_txt = open(self.target_tsv_path, "r", encoding="utf8").readlines()
            sentences_words: List[List[str]] = []
            sentences_labels: List[List[str]] = []
            sentences_labelled_entities: List[List[str]] = []
            sentences_labelled_entities_labels: List[List[str]] = []

            for line in lines_txt:
                line = line.strip()
                words = line.split()
                labels = ["O"] * len(words)
                sentences_words.append(words)
                sentences_labels.append(labels)
                sentences_labelled_entities.append([])
                sentences_labelled_entities_labels.append([])
            target_dataset_sentences = [
                sentences_words,
                sentences_labels,
                sentences_labelled_entities,
                sentences_labelled_entities_labels,
            ]
        else:
            target_dataset_sentences = read_all_sentences_tsv(
                self.target_tsv_path,
            )

        target_dataset_sentences = list(map(list, zip(*target_dataset_sentences)))

        self.dataset = []

        for (
            (
                i,
                (
                    source_words,
                    source_labels,
                    source_labelled_entities,
                    source_labelled_entities_labels,
                ),
            ),
            (
                target_words,
                target_labels,
                target_labelled_entities,
                target_labelled_entities_labels,
            ),
        ) in zip(
            enumerate(tqdm(source_dataset_sentences, desc="Tokenization")),
            target_dataset_sentences,
        ):
            x_source, y_source = format_sentence_with_labels(
                words=source_words,
                labelled_entities=source_labelled_entities,
                labelled_entities_labels=source_labelled_entities_labels,
            )

            x_target, y_target = format_sentence_with_labels(
                words=target_words,
                labelled_entities=target_labelled_entities
                if not inference
                else source_labelled_entities,
                labelled_entities_labels=target_labelled_entities_labels
                if not inference
                else source_labelled_entities_labels,
            )

            self.dataset.append(
                prepare_sl(
                    tokenizer=tokenizer,
                    x=x_target,
                    y=y_target,
                    source_words=" ".join(source_words),
                    source_entities=y_source,
                    source_labels=" ".join(source_labels),
                    target_words=" ".join(target_words),
                    target_entities=y_target if not inference else "",
                    target_labels=" ".join(target_labels),
                    task_name="SequenceLabelling",
                    idx=i,
                    max_source_len=self.max_source_len,
                    max_target_len=self.max_target_len,
                )
            )

        print(f"Dataset len: {len(self.dataset)} sentences")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def collate_fn_t5(data_collator_seq2seq, pad_fn, batch):
    batch = list(map(list, zip(*batch)))

    model_inputs = data_collator_seq2seq(batch[0])
    task_name = pad_fn(batch[1])["input_ids"]
    idx = torch.tensor(batch[2])
    source_words = pad_fn(batch[3])["input_ids"]
    source_entities = pad_fn(batch[4])["input_ids"]
    source_labels = pad_fn(batch[5])["input_ids"]
    target_words = pad_fn(batch[6])["input_ids"]
    target_entities = pad_fn(batch[7])["input_ids"]
    target_labels = pad_fn(batch[8])["input_ids"]

    return (
        model_inputs,
        task_name,
        idx,
        source_words,
        source_entities,
        source_labels,
        target_words,
        target_entities,
        target_labels,
    )


def get_dataloader_score(
    jsonl_path: str,
    tokenizer: PreTrainedTokenizer,
    source_lang: str,
    target_lang: str,
    decoder_start_token_id: int,  # model.config.decoder_start_token_id
    normalize: bool = True,
    both_directions: bool = True,
    max_length: int = 512,
    top_k: int = None,
    batch_size: int = 1,
    num_workers: int = min(os.cpu_count(), 16),
):
    dataset = ScoreDataset(
        jsonl_path=jsonl_path,
        tokenizer=tokenizer,
        source_lang=source_lang,
        target_lang=target_lang,
        decoder_start_token_id=decoder_start_token_id,
        normalize=normalize,
        both_directions=both_directions,
        max_length=max_length,
        top_k=top_k,
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        label_pad_token_id=tokenizer.pad_token_id,
        padding=True,
        pad_to_multiple_of=8,  # May be faster on some hardware
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=num_workers,
        shuffle=False,
    )

    return dataloader


def get_dataloader_t5(
    source_tsv_path: str,
    target_tsv_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    max_source_len: int = 128,
    max_target_len: int = 128,
    shuffle: bool = False,
    inference: bool = False,
):
    dataset = T5Dataset(
        source_tsv_path=source_tsv_path,
        target_tsv_path=target_tsv_path,
        tokenizer=tokenizer,
        max_source_len=max_source_len,
        max_target_len=max_target_len,
        inference=inference,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        padding=True,
        label_pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=8,  # May be faster on some hardware
    )

    pad_fn = partial(
        tokenizer.pad,
        return_tensors="pt",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn_t5, data_collator, pad_fn),
        pin_memory=True,
        num_workers=min(os.cpu_count(), 16),
    )

    return dataloader
