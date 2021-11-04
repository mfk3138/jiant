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

    @staticmethod
    def amr_elements_tokenize(tokenizer, amr_elements):
        element_sub_token_lists = [tokenizer.tokenize(element) for element in amr_elements]
        element_lengths = [len(sub_tokens) for sub_tokens in element_sub_token_lists]
        element_sub_tokens = sum(element_sub_token_lists, [])
        total_length = 0
        element_end_indices = [total_length + length for length in element_lengths]
        return element_sub_tokens, element_end_indices

    def tokenize(self, tokenizer):
        premise_concept_sub_tokens, premise_concept_end_indices = self.amr_elements_tokenize(tokenizer, self.premise_concepts)
        premise_relation_label_sub_tokens, premise_relation_label_end_indices = self.amr_elements_tokenize(tokenizer, self.premise_relation_labels)
        hypothesis_concept_sub_tokens, hypothesis_concept_end_indices = self.amr_elements_tokenize(tokenizer, self.hypothesis_concepts)
        hypothesis_relation_label_sub_tokens, hypothesis_relation_label_end_indices = self.amr_elements_tokenize(tokenizer, self.hypothesis_relation_labels)
        return TokenizedExample(
            guid=self.guid,
            premise_snt=tokenizer.tokenize(self.premise_snt),
            premise_concept_sub_tokens=premise_concept_sub_tokens,
            premise_concept_end_indices=premise_concept_end_indices,
            premise_relation_ids=self.premise_relation_ids,
            premise_relation_label_sub_tokens=premise_relation_label_sub_tokens,
            premise_relation_label_end_indices=premise_relation_label_end_indices,
            hypothesis_snt=tokenizer.tokenize(self.hypothesis_snt),
            hypothesis_concept_sub_tokens=hypothesis_concept_sub_tokens,
            hypothesis_concept_end_indices=hypothesis_concept_end_indices,
            hypothesis_relation_ids=self.hypothesis_relation_ids,
            hypothesis_relation_label_sub_tokens=hypothesis_relation_label_sub_tokens,
            hypothesis_relation_label_end_indices=hypothesis_relation_label_end_indices,
            label_id=MnliAMRTask.LABEL_TO_ID[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    premise_snt: List
    premise_concept_sub_tokens: List
    premise_concept_end_indices: List
    premise_relation_ids: List[Tuple[int, int]]
    premise_relation_label_sub_tokens: List
    premise_relation_label_end_indices: List
    hypothesis_snt: List
    hypothesis_concept_sub_tokens: List
    hypothesis_concept_end_indices: List
    hypothesis_relation_ids: List[Tuple[int, int]]
    hypothesis_relation_label_sub_tokens: List
    hypothesis_relation_label_end_indices: List
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        return double_sentence_with_amr_featurize(
            guid=self.guid,
            input_tokens_a=self.premise_snt,
            input_amr_concept_sub_tokens_a=self.premise_concept_sub_tokens,
            input_amr_concept_end_indices_a=self.premise_concept_end_indices,
            input_amr_relation_ids_a=self.premise_relation_ids,
            input_amr_relation_label_sub_tokens_a=self.premise_relation_label_sub_tokens,
            input_amr_relation_label_end_indices_a=self.premise_relation_label_end_indices,
            input_tokens_b=self.hypothesis_snt,
            input_amr_concept_sub_tokens_b=self.hypothesis_concept_sub_tokens,
            input_amr_concept_end_indices_b=self.hypothesis_concept_end_indices,
            input_amr_relation_ids_b=self.hypothesis_relation_ids,
            input_amr_relation_label_sub_tokens_b=self.hypothesis_relation_label_sub_tokens,
            input_amr_relation_label_end_indices_b=self.hypothesis_relation_label_end_indices,
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
    input_concept_sub_token_ids: np.ndarray
    input_concept_end_indices: np.ndarray
    input_concept_end_indices_mask: np.ndarray
    input_relation_ids: np.ndarray
    input_relation_label_sub_token_ids: np.ndarray
    input_relation_label_end_indices: np.ndarray
    input_relation_label_end_indices_mask: np.ndarray
    label_id: int
    tokens: list


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    input_concept_sub_token_ids: torch.LongTensor
    input_concept_end_indices: torch.LongTensor
    input_concept_end_indices_mask: torch.LongTensor
    input_relation_ids: torch.LongTensor
    input_relation_label_sub_token_ids: torch.LongTensor
    input_relation_label_end_indices: torch.LongTensor
    input_relation_label_end_indices_mask: torch.LongTensor
    label_id: torch.LongTensor
    tokens: list


class MnliAMRTask(GlueMixin, Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.CLASSIFICATION
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
