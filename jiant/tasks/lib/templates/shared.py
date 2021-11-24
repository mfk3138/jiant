import numpy as np
from dataclasses import dataclass
from typing import List, NamedTuple, Tuple

from jiant.tasks.core import FeaturizationSpec
from jiant.tasks.utils import truncate_sequences, pad_to_max_seq_length
from jiant.utils.python.datastructures import BiMap


MAX_SUB_TOKEN_LENGTH = 5
MAX_CONCEPT_LENGTH = 512
MAX_RELATION_LENGTH = 512


class Span(NamedTuple):
    start: int
    end: int  # Use exclusive end, for consistency

    def add(self, i: int):
        return Span(start=self.start + i, end=self.end + i)

    def to_slice(self):
        return slice(*self)

    def to_array(self):
        return np.array([self.start, self.end])


@dataclass
class UnpaddedInputs:
    unpadded_tokens: List
    unpadded_segment_ids: List
    cls_offset: int


@dataclass
class UnpaddedAMRInputs:
    unpadded_concepts: List[List[str]]
    unpadded_relation_ids: List[Tuple[int, int]]
    unpadded_relation_labels: List[List[str]]


@dataclass
class InputSet:
    input_ids: List
    input_mask: List
    segment_ids: List


@dataclass
class AMRInputSet:
    concept_sub_token_ids: List[List[int]]
    concept_sub_token_mask: List[List[int]]
    relation_ids: List[Tuple[int, int]]
    relation_id_mask: List[int]
    relation_label_sub_token_ids: List[List[int]]
    relation_label_sub_token_mask: List[List[int]]


def single_sentence_featurize(
        guid: str,
        input_tokens: List[str],
        label_id: int,
        tokenizer,
        feat_spec: FeaturizationSpec,
        data_row_class,
):
    unpadded_inputs = construct_single_input_tokens_and_segment_ids(
        input_tokens=input_tokens, tokenizer=tokenizer, feat_spec=feat_spec,
    )
    return create_generic_data_row_from_tokens_and_segments(
        guid=guid,
        unpadded_tokens=unpadded_inputs.unpadded_tokens,
        unpadded_segment_ids=unpadded_inputs.unpadded_segment_ids,
        label_id=label_id,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
        data_row_class=data_row_class,
    )


def double_sentence_featurize(
        guid: str,
        input_tokens_a: List[str],
        input_tokens_b: List[str],
        label_id: int,
        tokenizer,
        feat_spec: FeaturizationSpec,
        data_row_class,
):
    """Featurize an example for a two-input/two-sentence task, and return the example as a DataRow.

    Args:
        guid (str): human-readable identifier for interpretability and debugging.
        input_tokens_a (List[str]): sequence of tokens in segment a.
        input_tokens_b (List[str]): sequence of tokens in segment b.
        label_id (int): int representing the label for the task.
        tokenizer:
        feat_spec (FeaturizationSpec): Tokenization-related metadata.
        data_row_class (DataRow): DataRow class used in the task.

    Returns:
        DataRow representing an example.

    """
    unpadded_inputs = construct_double_input_tokens_and_segment_ids(
        input_tokens_a=input_tokens_a,
        input_tokens_b=input_tokens_b,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
    )

    return create_generic_data_row_from_tokens_and_segments(
        guid=guid,
        unpadded_tokens=unpadded_inputs.unpadded_tokens,
        unpadded_segment_ids=unpadded_inputs.unpadded_segment_ids,
        label_id=label_id,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
        data_row_class=data_row_class,
    )


def double_sentence_with_amr_featurize(
        guid: str,
        input_tokens_a: List[str],
        input_amr_concepts_a: List[List[str]],
        input_amr_relation_ids_a: List[Tuple[int, int]],
        input_amr_relation_labels_a: List[List[str]],
        input_tokens_b: List[str],
        input_amr_concepts_b: List[List[str]],
        input_amr_relation_ids_b: List[Tuple[int, int]],
        input_amr_relation_labels_b: List[List[str]],
        label_id: int,
        tokenizer,
        feat_spec: FeaturizationSpec,
        data_row_class,
):
    """Featurize an example for a two-input/two-sentence with AMR task, and return the example as a DataRow.

    Args:
        guid (str): human-readable identifier for interoperability and debugging.
        input_tokens_a (List[str]): sequence of tokens in segment a.
        input_amr_concepts_a (List[List[str]]): sequence of sub tokens of concepts in AMR a.
        input_amr_relation_ids_a (List[(int, int)]): sequence of (source, target)
            based on concept indices for relations in AMR a.
        input_amr_relation_labels_a (List[List[str]]): sequence of sub tokens of relation labels in AMR a.
        input_tokens_b (List[str]): sequence of tokens in segment b.
        input_amr_concepts_b (List[List[str]]): sequence of sub tokens of concepts in AMR b.
        input_amr_relation_ids_b (List[(int, int)]): sequence of (source, target)
            based on concept indices for relations in AMR b.
        input_amr_relation_labels_b (List[List[str]]): sequence of sub tokens of relation labels in AMR b.
        label_id (int): int representing the label for the task.
        tokenizer:
        feat_spec (FeaturizationSpec): Tokenization-related metadata.
        data_row_class (DataRow): DataRow class used in the task.

    Returns:
        DataRow representing an example.

    """
    unpadded_inputs = construct_double_input_tokens_and_segment_ids(
        input_tokens_a=input_tokens_a,
        input_tokens_b=input_tokens_b,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
    )

    unpadded_amr_inputs = construct_double_input_amr_concepts_and_relations(
        input_amr_concepts_a=input_amr_concepts_a,
        input_amr_relation_ids_a=input_amr_relation_ids_a,
        input_amr_relation_labels_a=input_amr_relation_labels_a,
        input_amr_concepts_b=input_amr_concepts_b,
        input_amr_relation_ids_b=input_amr_relation_ids_b,
        input_amr_relation_labels_b=input_amr_relation_labels_b,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
    )

    return create_generic_data_row_with_amr(
        guid=guid,
        unpadded_tokens=unpadded_inputs.unpadded_tokens,
        unpadded_segment_ids=unpadded_inputs.unpadded_segment_ids,
        unpadded_concepts=unpadded_amr_inputs.unpadded_concepts,
        unpadded_relation_ids=unpadded_amr_inputs.unpadded_relation_ids,
        unpadded_relation_labels=unpadded_amr_inputs.unpadded_relation_labels,
        label_id=label_id,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
        data_row_class=data_row_class,
    )


def construct_single_input_tokens_and_segment_ids(
        input_tokens: List[str], tokenizer, feat_spec: FeaturizationSpec
):
    special_tokens_count = 2  # CLS, SEP

    (input_tokens,) = truncate_sequences(
        tokens_ls=[input_tokens], max_length=feat_spec.max_seq_length - special_tokens_count,
    )

    return add_cls_token(
        unpadded_tokens=input_tokens + [tokenizer.sep_token],
        unpadded_segment_ids=(
                [feat_spec.sequence_a_segment_id]
                + [feat_spec.sequence_a_segment_id] * (len(input_tokens))
        ),
        tokenizer=tokenizer,
        feat_spec=feat_spec,
    )


def construct_double_input_tokens_and_segment_ids(
        input_tokens_a: List[str], input_tokens_b: List[str], tokenizer, feat_spec: FeaturizationSpec
):
    """Create token and segment id sequences, apply truncation, add separator and class tokens.

    Args:
        input_tokens_a (List[str]): sequence of tokens in segment a.
        input_tokens_b (List[str]): sequence of tokens in segment b.
        tokenizer:
        feat_spec (FeaturizationSpec): Tokenization-related metadata.

    Returns:
        UnpaddedInputs: unpadded inputs with truncation applied and special tokens appended.

    """
    if feat_spec.sep_token_extra:
        maybe_extra_sep = [tokenizer.sep_token]
        maybe_extra_sep_segment_id = [feat_spec.sequence_a_segment_id]
        special_tokens_count = 4  # CLS, SEP-SEP, SEP
    else:
        maybe_extra_sep = []
        maybe_extra_sep_segment_id = []
        special_tokens_count = 3  # CLS, SEP, SEP

    input_tokens_a, input_tokens_b = truncate_sequences(
        tokens_ls=[input_tokens_a, input_tokens_b],
        max_length=feat_spec.max_seq_length - special_tokens_count,
    )

    unpadded_tokens = (
            input_tokens_a
            + [tokenizer.sep_token]
            + maybe_extra_sep
            + input_tokens_b
            + [tokenizer.sep_token]
    )
    unpadded_segment_ids = (
            [feat_spec.sequence_a_segment_id] * len(input_tokens_a)
            + [feat_spec.sequence_a_segment_id]
            + maybe_extra_sep_segment_id
            + [feat_spec.sequence_b_segment_id] * len(input_tokens_b)
            + [feat_spec.sequence_b_segment_id]
    )
    return add_cls_token(
        unpadded_tokens=unpadded_tokens,
        unpadded_segment_ids=unpadded_segment_ids,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
    )


def construct_double_input_amr_concepts_and_relations(
        input_amr_concepts_a: List[List[str]],
        input_amr_relation_ids_a: List[Tuple[int, int]],
        input_amr_relation_labels_a: List[List[str]],
        input_amr_concepts_b: List[List[str]],
        input_amr_relation_ids_b: List[Tuple[int, int]],
        input_amr_relation_labels_b: List[List[str]],
        tokenizer,
        feat_spec: FeaturizationSpec,
):
    """ Merge concepts, relation ids and labels from 2 AMRs, apply truncation.

    Args:
        input_amr_concepts_a (List[List[str]]): sequence of sub tokens of concepts in AMR a.
        input_amr_relation_ids_a (List[(int, int)]):
            sequence of (source, target) based on concept indices for relations in AMR a.
        input_amr_relation_labels_a (List[List[str]]): sequence of sub tokens of relation labels in AMR a.
        input_amr_concepts_b (List[List[str]]): sequence of sub tokens of concepts in AMR b.
        input_amr_relation_ids_b (List[(int, int)]):
            sequence of (source, target) based on concept indices for relations in AMR b.
        input_amr_relation_labels_b (List[List[str]]): sequence of sub tokens of relation labels in AMR b.
        tokenizer:
        feat_spec (FeaturizationSpec): Tokenization-related metadata.

    Returns:
        UnpaddedAMRInputs: unpadded merged AMR inputs.

    """
    # TODO: 1、sub token长度裁剪；2、concepts长度裁剪，相应修改relation ids和labels；3、合并，相应修改relation ids
    input_amr_concepts_a = sum([truncate_sequences(tokens_ls=[concept], max_length=MAX_SUB_TOKEN_LENGTH)
                            for concept in input_amr_concepts_a], [])
    input_amr_concepts_b = sum([truncate_sequences(tokens_ls=[concept], max_length=MAX_SUB_TOKEN_LENGTH)
                            for concept in input_amr_concepts_b], [])
    input_amr_relation_labels_a = sum([truncate_sequences(tokens_ls=[label], max_length=MAX_SUB_TOKEN_LENGTH)
                            for label in input_amr_relation_labels_a], [])
    input_amr_relation_labels_b = sum([truncate_sequences(tokens_ls=[label], max_length=MAX_SUB_TOKEN_LENGTH)
                            for label in input_amr_relation_labels_b], [])
    input_amr_concepts_a, input_amr_concepts_b = truncate_sequences(
        tokens_ls=[input_amr_concepts_a, input_amr_concepts_b], max_length=MAX_CONCEPT_LENGTH)
    truncate_input_amr_relation_ids_a = []
    truncate_input_amr_relation_labels_a = []
    truncate_input_amr_relation_ids_b = []
    truncate_input_amr_relation_labels_b= []
    length_a = len(input_amr_concepts_a)
    length_b = len(input_amr_concepts_b)
    for relation_id, relation_label in zip(input_amr_relation_ids_a, input_amr_relation_labels_a):
        source, target = relation_id
        if source < length_a and target < length_a:
            truncate_input_amr_relation_ids_a.append(relation_id)
            truncate_input_amr_relation_labels_a.append(relation_label)
    for relation_id, relation_label in zip(input_amr_relation_ids_b, input_amr_relation_labels_b):
        source, target = relation_id
        if source < length_b and target < length_b:
            truncate_input_amr_relation_ids_b.append([source + length_a, target + length_a])
            truncate_input_amr_relation_labels_b.append(relation_label)
    truncate_input_amr_relation_ids_a, truncate_input_amr_relation_ids_b = truncate_sequences(
        tokens_ls=[truncate_input_amr_relation_ids_a, truncate_input_amr_relation_ids_b],
        max_length=MAX_RELATION_LENGTH)
    truncate_input_amr_relation_labels_a, truncate_input_amr_relation_labels_b = truncate_sequences(
        tokens_ls=[truncate_input_amr_relation_labels_a, truncate_input_amr_relation_labels_b],
        max_length=MAX_RELATION_LENGTH)
    return UnpaddedAMRInputs(
        unpadded_concepts=input_amr_concepts_a + input_amr_concepts_b,
        unpadded_relation_ids=truncate_input_amr_relation_ids_a + truncate_input_amr_relation_ids_b,
        unpadded_relation_labels=truncate_input_amr_relation_labels_a + truncate_input_amr_relation_labels_b,
    )


def add_cls_token(
        unpadded_tokens: List[str],
        unpadded_segment_ids: List[int],
        tokenizer,
        feat_spec: FeaturizationSpec,
):
    """Add class token to unpadded inputs.

    Applies class token to end (or start) of unpadded inputs depending on FeaturizationSpec.

    Args:
        unpadded_tokens (List[str]): sequence of unpadded token strings.
        unpadded_segment_ids (List[str]): sequence of unpadded segment ids.
        tokenizer:
        feat_spec (FeaturizationSpec): Tokenization-related metadata.

    Returns:
        UnpaddedInputs: unpadded inputs with class token appended.

    """
    if feat_spec.cls_token_at_end:
        return UnpaddedInputs(
            unpadded_tokens=unpadded_tokens + [tokenizer.cls_token],
            unpadded_segment_ids=unpadded_segment_ids + [feat_spec.cls_token_segment_id],
            cls_offset=0,
        )
    else:
        return UnpaddedInputs(
            unpadded_tokens=[tokenizer.cls_token] + unpadded_tokens,
            unpadded_segment_ids=[feat_spec.cls_token_segment_id] + unpadded_segment_ids,
            cls_offset=1,
        )


def create_generic_data_row_from_tokens_and_segments(
        guid: str,
        unpadded_tokens: List[str],
        unpadded_segment_ids: List[int],
        label_id: int,
        tokenizer,
        feat_spec: FeaturizationSpec,
        data_row_class,
):
    """Creates an InputSet and wraps the InputSet into a DataRow class.

    Args:
        guid (str): human-readable identifier (for interpretability and debugging).
        unpadded_tokens (List[str]): sequence of unpadded token strings.
        unpadded_segment_ids (List[int]): sequence of unpadded segment ids.
        label_id (int): int representing the label for the task.
        tokenizer:
        feat_spec (FeaturizationSpec): Tokenization-related metadata.
        data_row_class (DataRow): data row class to wrap and return the inputs.

    Returns:
        DataRow: data row class containing model inputs.

    """
    input_set = create_input_set_from_tokens_and_segments(
        unpadded_tokens=unpadded_tokens,
        unpadded_segment_ids=unpadded_segment_ids,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
    )
    return data_row_class(
        guid=guid,
        input_ids=np.array(input_set.input_ids),
        input_mask=np.array(input_set.input_mask),
        segment_ids=np.array(input_set.segment_ids),
        label_id=label_id,
        tokens=unpadded_tokens,
    )


def create_generic_data_row_with_amr(
        guid: str,
        unpadded_tokens: List[str],
        unpadded_segment_ids: List[int],
        unpadded_concepts: List[List[str]],
        unpadded_relation_ids: List[Tuple[int, int]],
        unpadded_relation_labels: List[List[str]],
        label_id: int,
        tokenizer,
        feat_spec: FeaturizationSpec,
        data_row_class,
):
    """Creates an InputSet and wraps the InputSet into a DataRow class.

    Args:
        guid (str): human-readable identifier (for interpretability and debugging).
        unpadded_tokens (List[str]): sequence of unpadded token strings.
        unpadded_segment_ids (List[int]): sequence of unpadded segment ids.
        unpadded_concepts (List[List[str]]): sequence of unpadded sub tokens of AMR concepts.
        unpadded_relation_ids (List[(int, int)]): sequence of unpadded (source, target)
            based on concept indices for AMR relations.
        unpadded_relation_labels (List[List[str]]): sequence of unpadded sub tokens of AMR relation labels.
        label_id (int): int representing the label for the task.
        tokenizer:
        feat_spec (FeaturizationSpec): Tokenization-related metadata.
        data_row_class (DataRow): data row class to wrap and return the inputs.

    Returns:
        DataRow: data row class containing model inputs.

    """
    input_set = create_input_set_from_tokens_and_segments(
        unpadded_tokens=unpadded_tokens,
        unpadded_segment_ids=unpadded_segment_ids,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
    )
    amr_input_set = create_amr_input_set(
        unpadded_concepts=unpadded_concepts,
        unpadded_relation_ids=unpadded_relation_ids,
        unpadded_relation_labels=unpadded_relation_labels,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
    )
    return data_row_class(
        guid=guid,
        input_ids=np.array(input_set.input_ids),
        input_mask=np.array(input_set.input_mask),
        segment_ids=np.array(input_set.segment_ids),
        input_concept_ids=np.array(amr_input_set.concept_sub_token_ids),
        input_concept_mask=np.array(amr_input_set.concept_sub_token_mask),
        input_relation_ids=np.array(amr_input_set.relation_ids),
        input_relation_id_mask=np.array(amr_input_set.relation_id_mask),
        input_relation_label_ids=np.array(amr_input_set.relation_label_sub_token_ids),
        input_relation_label_mask=np.array(amr_input_set.relation_label_sub_token_mask),
        label_id=label_id,
        tokens=unpadded_tokens,
    )


def create_input_set_from_tokens_and_segments(
        unpadded_tokens: List[str],
        unpadded_segment_ids: List[int],
        tokenizer,
        feat_spec: FeaturizationSpec,
):
    """Create padded inputs for model.

    Converts tokens to ids, makes input set (input ids, input mask, and segment ids), adds padding.

    Args:
        unpadded_tokens (List[str]): unpadded list of token strings.
        unpadded_segment_ids (List[int]): unpadded list of segment ids.
        tokenizer:
        feat_spec (FeaturizationSpec): Tokenization-related metadata.

    Returns:
        Padded input set.

    """
    assert len(unpadded_tokens) == len(unpadded_segment_ids)
    input_ids = tokenizer.convert_tokens_to_ids(unpadded_tokens)
    input_mask = [1] * len(input_ids)
    input_set = pad_features_with_feat_spec(
        input_ids=input_ids,
        input_mask=input_mask,
        unpadded_segment_ids=unpadded_segment_ids,
        feat_spec=feat_spec,
    )
    return input_set


def create_amr_input_set(
        unpadded_concepts: List[List[str]],
        unpadded_relation_ids: List[Tuple[int, int]],
        unpadded_relation_labels: List[List[str]],
        tokenizer,
        feat_spec: FeaturizationSpec,
):
    """Create padded inputs for model.

    Converts tokens to ids, makes input set (input ids, input mask, and segment ids), adds padding.

    Args:
        unpadded_tokens (List[str]): unpadded list of token strings.
        unpadded_segment_ids (List[int]): unpadded list of segment ids.
        tokenizer:
        feat_spec (FeaturizationSpec): Tokenization-related metadata.

    Returns:
        Padded amr input set.

    """
    # TODO：1、convert_tokens_to_ids；2、对concepts和relation labels进行二维padding，并生成mask
    assert len(unpadded_relation_ids) == len(unpadded_relation_labels)
    concept_sub_token_ids = [tokenizer.convert_tokens_to_ids(concept_sub_tokens)
                             for concept_sub_tokens in unpadded_concepts]
    relation_label_sub_token_ids = [tokenizer.convert_tokens_to_ids(relation_label_sub_tokens)
                                    for relation_label_sub_tokens in unpadded_relation_labels]
    concept_sub_token_mask = [pad_to_max_seq_length(ls=[1] * len(sub_tokens),
                                                    max_seq_length=MAX_SUB_TOKEN_LENGTH,
                                                    pad_right=not feat_spec.pad_on_left)
                              for sub_tokens in concept_sub_token_ids]
    concept_sub_token_ids = [pad_to_max_seq_length(ls=sub_tokens,
                                                   max_seq_length=MAX_SUB_TOKEN_LENGTH,
                                                   pad_idx=feat_spec.pad_token_id,
                                                   pad_right=not feat_spec.pad_on_left)
                             for sub_tokens in concept_sub_token_ids]
    concept_sub_token_mask = pad_to_max_seq_length(ls=concept_sub_token_mask,
                                                   max_seq_length=feat_spec.max_seq_length,
                                                   pad_idx=[0] * MAX_SUB_TOKEN_LENGTH,
                                                   pad_right=not feat_spec.pad_on_left)
    concept_sub_token_ids = pad_to_max_seq_length(ls=concept_sub_token_ids,
                                                  max_seq_length=feat_spec.max_seq_length,
                                                  pad_idx=[feat_spec.pad_token_id] * MAX_SUB_TOKEN_LENGTH,
                                                  pad_right=not feat_spec.pad_on_left)
    relation_id_mask = pad_to_max_seq_length(ls=[1] * len(unpadded_relation_ids),
                                             max_seq_length=feat_spec.max_seq_length,
                                             pad_right=not feat_spec.pad_on_left)
    relation_ids = pad_to_max_seq_length(ls=unpadded_relation_ids,
                                         max_seq_length=feat_spec.max_seq_length,
                                         pad_idx=[-1, -1],
                                         pad_right=not feat_spec.pad_on_left)
    relation_label_sub_token_mask = [pad_to_max_seq_length(ls=[1] * len(sub_tokens),
                                                           max_seq_length=MAX_SUB_TOKEN_LENGTH,
                                                           pad_right=not feat_spec.pad_on_left)
                                     for sub_tokens in relation_label_sub_token_ids]
    relation_label_sub_token_ids = [pad_to_max_seq_length(ls=sub_tokens,
                                                          max_seq_length=MAX_SUB_TOKEN_LENGTH,
                                                          pad_idx=feat_spec.pad_token_id,
                                                          pad_right=not feat_spec.pad_on_left)
                                    for sub_tokens in relation_label_sub_token_ids]
    relation_label_sub_token_mask = pad_to_max_seq_length(ls=relation_label_sub_token_mask,
                                                          max_seq_length=feat_spec.max_seq_length,
                                                          pad_idx=[0] * MAX_SUB_TOKEN_LENGTH,
                                                          pad_right=not feat_spec.pad_on_left)
    relation_label_sub_token_ids = pad_to_max_seq_length(ls=relation_label_sub_token_ids,
                                                         max_seq_length=feat_spec.max_seq_length,
                                                         pad_idx=[feat_spec.pad_token_id] * MAX_SUB_TOKEN_LENGTH,
                                                         pad_right=not feat_spec.pad_on_left)
    return AMRInputSet(
        concept_sub_token_ids=concept_sub_token_ids,
        concept_sub_token_mask=concept_sub_token_mask,
        relation_ids=relation_ids,
        relation_id_mask=relation_id_mask,
        relation_label_sub_token_ids=relation_label_sub_token_ids,
        relation_label_sub_token_mask=relation_label_sub_token_mask,
    )


def pad_features_with_feat_spec(
        input_ids: List[int],
        input_mask: List[int],
        unpadded_segment_ids: List[int],
        feat_spec: FeaturizationSpec,
):
    """Apply padding to feature set according to settings from FeaturizationSpec.

    Args:
        input_ids (List[int]): sequence unpadded input ids.
        input_mask (List[int]): unpadded input mask sequence.
        unpadded_segment_ids (List[int]): sequence of unpadded segment ids.
        feat_spec (FeaturizationSpec): Tokenization-related metadata.

    Returns:
        InputSet: input set containing padded input ids, input mask, and segment ids.

    """
    return InputSet(
        input_ids=pad_single_with_feat_spec(
            ls=input_ids, feat_spec=feat_spec, pad_idx=feat_spec.pad_token_id,
        ),
        input_mask=pad_single_with_feat_spec(
            ls=input_mask, feat_spec=feat_spec, pad_idx=feat_spec.pad_token_mask_id,
        ),
        segment_ids=pad_single_with_feat_spec(
            ls=unpadded_segment_ids, feat_spec=feat_spec, pad_idx=feat_spec.pad_token_segment_id,
        ),
    )


def pad_single_with_feat_spec(
        ls: List[int], feat_spec: FeaturizationSpec, pad_idx: int, check=True
):
    """Apply padding to sequence according to settings from FeaturizationSpec.

    Args:
        ls (List[int]): sequence to pad.
        feat_spec (FeaturizationSpec): metadata containing max sequence length and padding settings.
        pad_idx (int): element to use for padding.
        check (bool): True if padded length should be checked as under the max sequence length.

    Returns:
        Sequence with padding applied.

    """
    return pad_to_max_seq_length(
        ls=ls,
        max_seq_length=feat_spec.max_seq_length,
        pad_idx=pad_idx,
        pad_right=not feat_spec.pad_on_left,
        check=check,
    )


def labels_to_bimap(labels):
    """Creates mappings from label to id, and from id to label. See details in docs for BiMap.

    Args:
        labels: sequence of label to map to ids.

    Returns:
        Tuple[Dict, Dict]: mappings from labels to ids, and ids to labels.

    """
    label2id, id2label = BiMap(a=labels, b=list(range(len(labels)))).get_maps()
    return label2id, id2label
