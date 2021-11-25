import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple

from jiant.tasks.core import (
    BaseExample,
    BaseTokenizedExample,
    BaseDataRow,
    BatchMixin,
    GlueMixin,
    Task,
    TaskTypes,
)
from jiant.tasks.lib.templates.shared import double_sentence_with_amr_featurize, labels_to_bimap
from jiant.utils.python.io import read_jsonl


@dataclass
class Example(BaseExample):
    guid: str
    premise_snt: str
    premise_concepts: List[str]
    premise_relation_ids: List[Tuple[int, int]]
    premise_relation_labels: List[str]
    hypothesis_snt: str
    hypothesis_concepts: List[str]
    hypothesis_relation_ids: List[Tuple[int, int]]
    hypothesis_relation_labels: List[str]
    label: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            premise_snt=tokenizer.tokenize(self.premise_snt),
            premise_concepts=[tokenizer.tokenize(concept) for concept in self.premise_concepts],
            premise_relation_ids=self.premise_relation_ids,
            premise_relation_labels=[tokenizer.tokenize(relation_label)
                                     for relation_label in self.premise_relation_labels],
            hypothesis_snt=tokenizer.tokenize(self.hypothesis_snt),
            hypothesis_concepts=[tokenizer.tokenize(concept) for concept in self.hypothesis_concepts],
            hypothesis_relation_ids=self.hypothesis_relation_ids,
            hypothesis_relation_labels=[tokenizer.tokenize(relation_label)
                                        for relation_label in self.hypothesis_relation_labels],
            label_id=MnliAMRTask.LABEL_TO_ID[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    premise_snt: List[str]
    premise_concepts: List[List[str]]
    premise_relation_ids: List[Tuple[int, int]]
    premise_relation_labels: List[List[str]]
    hypothesis_snt: List[str]
    hypothesis_concepts: List[List[str]]
    hypothesis_relation_ids: List[Tuple[int, int]]
    hypothesis_relation_labels: List[List[str]]
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        return double_sentence_with_amr_featurize(
            guid=self.guid,
            input_tokens_a=self.premise_snt,
            input_amr_concepts_a=self.premise_concepts,
            input_amr_relation_ids_a=self.premise_relation_ids,
            input_amr_relation_labels_a=self.premise_relation_labels,
            input_tokens_b=self.hypothesis_snt,
            input_amr_concepts_b=self.hypothesis_concepts,
            input_amr_relation_ids_b=self.hypothesis_relation_ids,
            input_amr_relation_labels_b=self.hypothesis_relation_labels,
            label_id=self.label_id,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
            data_row_class=DataRow,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    input_concept_ids: np.ndarray
    input_concept_mask: np.ndarray
    input_relation_ids: np.ndarray
    input_relation_id_mask: np.ndarray
    input_relation_label_ids: np.ndarray
    input_relation_label_mask: np.ndarray
    label_id: int
    tokens: list


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    input_concept_ids: torch.LongTensor
    input_concept_mask: torch.LongTensor
    input_relation_ids: torch.LongTensor
    input_relation_id_mask: torch.LongTensor
    input_relation_label_ids: torch.LongTensor
    input_relation_label_mask: torch.LongTensor
    label_id: torch.LongTensor
    tokens: list


class MnliAMRTask(GlueMixin, Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.CLASSIFICATION_AMR
    LABELS = ["contradiction", "entailment", "neutral"]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    def get_train_examples(self):
        return self._create_examples(lines=read_jsonl(self.train_path), set_type="train")

    def get_val_examples(self):
        return self._create_examples(lines=read_jsonl(self.val_path), set_type="val")

    def get_test_examples(self):
        return self._create_examples(lines=read_jsonl(self.test_path), set_type="test")

    @classmethod
    def _create_examples(cls, lines, set_type):
        # noinspection DuplicatedCode
        examples = []
        for (i, line) in enumerate(lines):
            examples.append(
                Example(
                    # NOTE: get_glue_preds() is dependent on this guid format.
                    guid="%s-%s" % (set_type, i),
                    premise_snt=line["premise"]["snt"],
                    premise_concepts=line["premise"]["concepts"],
                    premise_relation_ids=line["premise"]["relation_ids"],
                    premise_relation_labels=line["premise"]["relation_labels"],
                    hypothesis_snt=line["hypothesis"]["snt"],
                    hypothesis_concepts=line["hypothesis"]["concepts"],
                    hypothesis_relation_ids=line["hypothesis"]["relation_ids"],
                    hypothesis_relation_labels=line["hypothesis"]["relation_labels"],
                    label=line["label"] if set_type != "test" else cls.LABELS[-1],
                )
            )
        return examples
