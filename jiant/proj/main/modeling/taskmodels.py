import abc

import torch
import torch.nn as nn

from typing import Callable

import jiant.proj.main.modeling.heads as heads
from develop.model.graph_encoder import RelationalTransformerEncoderLayer, RelationalTransformerEncoder

from jiant.proj.main.components.outputs import LogitsAndLossOutput
from jiant.proj.main.components.outputs import LogitsOutput
from jiant.utils.python.datastructures import take_one

from jiant.tasks.core import TaskTypes


class JiantTaskModelFactory:
    """This factory is used to create task models bundling the task,
       encoder, and task head within the task model.

    Attributes:
        registry (dict): Dynamic registry mapping task types to task models
    """

    registry = {}

    @classmethod
    def register(cls, task_type: TaskTypes) -> Callable:
        """Register task_type as a key mapping to a TaskModel

        Args:
            task_type (TaskTypes): TaskType key mapping to a BaseHead task head

        Returns:
            Callable: inner_wrapper() wrapping TaskModel constructor
        """

        def inner_wrapper(wrapped_class: Taskmodel) -> Callable:
            assert task_type not in cls.registry
            cls.registry[task_type] = wrapped_class
            return wrapped_class

        return inner_wrapper

    def __call__(cls, task, encoder, head, **kwargs):
        """This creates the TaskModel corresponding to the Task, abc.abstractmethod,
            and encoder used.

        Args:
            task (Task): Task
            encoder (JiantTransformersModel): encoder
            head (BaseHead): Task head
            **kwargs: Additional arguments for initializing TaskModel

        Returns:
            TaskModel: Initialized task model bundling task, encoder, and head
        """
        taskmodel_class = cls.registry[task.TASK_TYPE]
        taskmodel = taskmodel_class(task, encoder, head, **kwargs)
        return taskmodel


class Taskmodel(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, task, encoder, head):
        super().__init__()
        self.task = task
        self.encoder = encoder
        self.head = head

    def forward(self, batch, tokenizer, compute_loss: bool = False):
        raise NotImplementedError


@JiantTaskModelFactory.register(TaskTypes.CLASSIFICATION)
class ClassificationModel(Taskmodel):
    def __init__(self, task, encoder, head: heads.ClassificationHead, **kwargs):

        super().__init__(task=task, encoder=encoder, head=head)

    def forward(self, batch, tokenizer, compute_loss: bool = False):
        encoder_output = self.encoder.encode(
            input_ids=batch.input_ids, segment_ids=batch.segment_ids, input_mask=batch.input_mask,
        )
        logits = self.head(pooled=encoder_output.pooled)
        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.head.num_labels), batch.label_id.view(-1),)
            return LogitsAndLossOutput(logits=logits, loss=loss, other=encoder_output.other)
        else:
            return LogitsOutput(logits=logits, other=encoder_output.other)


@JiantTaskModelFactory.register(TaskTypes.CLASSIFICATION_AMR)
class ClassificationAMRModel(Taskmodel):
    def __init__(self, task, encoder, head: heads.ClassificationHead, **kwargs):

        super().__init__(task=task, encoder=encoder, head=head)
        encoder_layer = RelationalTransformerEncoderLayer(add_relation=True, d_model=768, nhead=8,
                                                          relation_type=kwargs["taskmodel_kwargs"]["relation_type"])
        self.graph_encoder = RelationalTransformerEncoder(encoder_layer, num_layers=6)
        self.fusion_type = kwargs["taskmodel_kwargs"]["fusion_type"]
        if self.fusion_type in [1, 2]:
            from torch.nn import TransformerDecoder, TransformerDecoderLayer
            decoder_layer = TransformerDecoderLayer(d_model=768, nhead=8)
            self.cross_attention_mixer = TransformerDecoder(decoder_layer, num_layers=1)

    def forward(self, batch, tokenizer, compute_loss: bool = False):
        concept_sub_ids = batch.input_concept_ids
        concept_sub_mask = batch.input_concept_mask
        relation_ids = batch.input_relation_ids
        relation_id_mask = batch.input_relation_id_mask
        relation_label_sub_ids = batch.input_relation_label_ids
        relation_label_sub_mask = batch.input_relation_label_mask
        # word embedding, sub token pooling and relation preparing
        bsz, length, sub_length = concept_sub_ids.size()
        pad_embedding = self.encoder.embeddings(concept_sub_ids.new(1, 1).fill_(tokenizer.pad_token_id)).squeeze()
        concept_embeddings = self.encoder.embeddings(concept_sub_ids.view(bsz * length, sub_length))\
            .view(bsz, length, sub_length, -1)
        concepts, _ = (concept_embeddings * concept_sub_mask.unsqueeze(-1).expand(concept_embeddings.size())).max(dim=2)
        concept_mask, _ = concept_sub_mask.max(dim=2)
        relation_label_embeddings = self.encoder.embeddings(relation_label_sub_ids.view(bsz * length, sub_length))\
            .view(bsz, length, sub_length, -1)
        relation_labels, _ = (relation_label_embeddings * relation_label_sub_mask.unsqueeze(-1)
                              .expand(relation_label_embeddings.size())).max(dim=2)
        relation_label_mask, _ = relation_label_sub_mask.max(dim=2)
        relation_labels = relation_labels + pad_embedding.view(1, 1, -1).expand(relation_labels.size()) \
                          * (1 - relation_label_mask).unsqueeze(-1).expand(relation_labels.size())
        batch_index = torch.arange(0, bsz).view(bsz, 1).type_as(relation_ids)
        # graph encoder
        concepts = concepts.permute(1, 0, 2)
        relation_dict = {"relation_labels": relation_labels,
                         "relation_ids": relation_ids,
                         "pad_embedding": pad_embedding,
                         "batch_index": batch_index}
        graph_features = self.graph_encoder(concepts, relation_dict, src_key_padding_mask=(1 - concept_mask).bool())
        # sentence encoder
        encoder_output = self.encoder.encode(
            input_ids=batch.input_ids, segment_ids=batch.segment_ids, input_mask=batch.input_mask,
        )
        # future fusion
        if self.fusion_type == 0:
            graph_features_pooled, _ = graph_features.max(0)
            fusion_features = torch.cat([graph_features_pooled, encoder_output.pooled], 1)
        elif self.fusion_type == 1:
            fusion_features = self.cross_attention_mixer(encoder_output.unpooled.transpose(0, 1),
                                                         graph_features,
                                                         tgt_key_padding_mask=batch.input_mask.bool(),
                                                         memory_key_padding_mask=(1 - concept_mask).bool())
            fusion_features, _ = fusion_features.max(0)
        elif self.fusion_type == 2:
            fusion_features = self.cross_attention_mixer(graph_features,
                                                         encoder_output.unpooled.transpose(0, 1),
                                                         tgt_key_padding_mask=(1 - concept_mask).bool(),
                                                         memory_key_padding_mask=batch.input_mask.bool())
            fusion_features, _ = fusion_features.max(0)
        else:
            raise Exception("Unsupported fusion type!")
        logits = self.head(pooled=fusion_features)
        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.head.num_labels), batch.label_id.view(-1),)
            return LogitsAndLossOutput(logits=logits, loss=loss, other=encoder_output.other)
        else:
            return LogitsOutput(logits=logits, other=encoder_output.other)


@JiantTaskModelFactory.register(TaskTypes.REGRESSION)
class RegressionModel(Taskmodel):
    def __init__(self, task, encoder, head: heads.RegressionHead, **kwargs):
        super().__init__(task=task, encoder=encoder, head=head)

    def forward(self, batch, tokenizer, compute_loss: bool = False):
        encoder_output = self.encoder.encode(
            input_ids=batch.input_ids, segment_ids=batch.segment_ids, input_mask=batch.input_mask,
        )
        # TODO: Abuse of notation - these aren't really logits  (issue #1187)
        logits = self.head(pooled=encoder_output.pooled)
        if compute_loss:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), batch.label.view(-1))
            return LogitsAndLossOutput(logits=logits, loss=loss, other=encoder_output.other)
        else:
            return LogitsOutput(logits=logits, other=encoder_output.other)


@JiantTaskModelFactory.register(TaskTypes.MULTIPLE_CHOICE)
class MultipleChoiceModel(Taskmodel):
    def __init__(self, task, encoder, head: heads.RegressionHead, **kwargs):
        super().__init__(task=task, encoder=encoder, head=head)
        self.num_choices = task.NUM_CHOICES

    def forward(self, batch, tokenizer, compute_loss: bool = False):
        choice_score_list = []
        encoder_output_other_ls = []
        for i in range(self.num_choices):
            encoder_output = self.encoder.encode(
                input_ids=batch.input_ids[:, i],
                segment_ids=batch.segment_ids[:, i],
                input_mask=batch.input_mask[:, i],
            )
            choice_score = self.head(pooled=encoder_output.pooled)
            choice_score_list.append(choice_score)
            encoder_output_other_ls.append(encoder_output.other)

        reshaped_outputs = []
        if encoder_output_other_ls[0]:
            for j in range(len(encoder_output_other_ls[0])):
                reshaped_outputs.append(
                    [
                        torch.stack([misc[j][layer_i] for misc in encoder_output_other_ls], dim=1,)
                        for layer_i in range(len(encoder_output_other_ls[0][0]))
                    ]
                )
            reshaped_outputs = tuple(reshaped_outputs)

        logits = torch.cat(
            [choice_score.unsqueeze(1).squeeze(-1) for choice_score in choice_score_list], dim=1
        )

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_choices), batch.label_id.view(-1))
            return LogitsAndLossOutput(logits=logits, loss=loss, other=reshaped_outputs)
        else:
            return LogitsOutput(logits=logits, other=reshaped_outputs)


@JiantTaskModelFactory.register(TaskTypes.SPAN_COMPARISON_CLASSIFICATION)
class SpanComparisonModel(Taskmodel):
    def __init__(self, task, encoder, head: heads.SpanComparisonHead, **kwargs):
        super().__init__(task=task, encoder=encoder, head=head)

    def forward(self, batch, tokenizer, compute_loss: bool = False):
        """Summary

        Args:
            batch (TYPE): Description
            tokenizer (TYPE): Description
            compute_loss (bool, optional): Description

        Returns:
            TYPE: Description
        """
        encoder_output = self.encoder.encode(
            input_ids=batch.input_ids, segment_ids=batch.segment_ids, input_mask=batch.input_mask,
        )
        logits = self.head(unpooled=encoder_output.unpooled, spans=batch.spans)
        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.head.num_labels), batch.label_id.view(-1),)
            return LogitsAndLossOutput(logits=logits, loss=loss, other=encoder_output.other)
        else:
            return LogitsOutput(logits=logits, other=encoder_output.other)


@JiantTaskModelFactory.register(TaskTypes.SPAN_PREDICTION)
class SpanPredictionModel(Taskmodel):
    def __init__(self, task, encoder, head: heads.TokenClassificationHead, **kwargs):
        super().__init__(task=task, encoder=encoder, head=head)
        self.offset_margin = 1000
        # 1000 is a big enough number that exp(-1000) will be strict 0 in float32.
        # So that if we add 1000 to the valid dimensions in the input of softmax,
        # we can guarantee the output distribution will only be non-zero at those dimensions.

    def forward(self, batch, tokenizer, compute_loss: bool = False):
        encoder_output = self.encoder.encode(
            input_ids=batch.input_ids, segment_ids=batch.segment_ids, input_mask=batch.input_mask,
        )
        logits = self.head(unpooled=encoder_output.unpooled)
        # Ensure logits in valid range is at least self.offset_margin higher than others
        logits_offset = logits.max() - logits.min() + self.offset_margin
        logits = logits + logits_offset * batch.selection_token_mask.unsqueeze(dim=2)
        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.transpose(dim0=1, dim1=2).flatten(end_dim=1), batch.gt_span_idxs.flatten(),
            )
            return LogitsAndLossOutput(logits=logits, loss=loss, other=encoder_output.other)
        else:
            return LogitsOutput(logits=logits, other=encoder_output.other)


@JiantTaskModelFactory.register(TaskTypes.MULTI_LABEL_SPAN_CLASSIFICATION)
class MultiLabelSpanComparisonModel(Taskmodel):
    def __init__(self, task, encoder, head: heads.SpanComparisonHead, **kwargs):
        super().__init__(task=task, encoder=encoder, head=head)

    def forward(self, batch, tokenizer, compute_loss: bool = False):
        encoder_output = self.encoder.encode(
            input_ids=batch.input_ids, segment_ids=batch.segment_ids, input_mask=batch.input_mask,
        )
        logits = self.head(unpooled=encoder_output.unpooled, spans=batch.spans)
        if compute_loss:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.head.num_labels), batch.label_ids.float(),)
            return LogitsAndLossOutput(logits=logits, loss=loss, other=encoder_output.other)
        else:
            return LogitsOutput(logits=logits, other=encoder_output.other)


@JiantTaskModelFactory.register(TaskTypes.TAGGING)
class TokenClassificationModel(Taskmodel):
    """From RobertaForTokenClassification"""

    def __init__(self, task, encoder, head: heads.TokenClassificationHead, **kwargs):
        super().__init__(task=task, encoder=encoder, head=head)

    def forward(self, batch, tokenizer, compute_loss: bool = False):
        encoder_output = self.encoder.encode(
            input_ids=batch.input_ids, segment_ids=batch.segment_ids, input_mask=batch.input_mask,
        )
        logits = self.head(unpooled=encoder_output.unpooled)
        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = batch.label_mask.view(-1) == 1
            active_logits = logits.view(-1, self.head.num_labels)[active_loss]
            active_labels = batch.label_ids.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return LogitsAndLossOutput(logits=logits, loss=loss, other=encoder_output.other)
        else:
            return LogitsOutput(logits=logits, other=encoder_output.other)


@JiantTaskModelFactory.register(TaskTypes.SQUAD_STYLE_QA)
class QAModel(Taskmodel):
    def __init__(self, task, encoder, head: heads.QAHead, **kwargs):
        super().__init__(task=task, encoder=encoder, head=head)

    def forward(self, batch, tokenizer, compute_loss: bool = False):
        encoder_output = self.encoder.encode(
            input_ids=batch.input_ids, segment_ids=batch.segment_ids, input_mask=batch.input_mask,
        )
        logits = self.head(unpooled=encoder_output.unpooled)
        if compute_loss:
            loss = compute_qa_loss(
                logits=logits,
                start_positions=batch.start_position,
                end_positions=batch.end_position,
            )
            return LogitsAndLossOutput(logits=logits, loss=loss, other=encoder_output.other)
        else:
            return LogitsOutput(logits=logits, other=encoder_output.other)


@JiantTaskModelFactory.register(TaskTypes.MASKED_LANGUAGE_MODELING)
class MLMModel(Taskmodel):
    def __init__(self, task, encoder, head: heads.BaseMLMHead, **kwargs):
        super().__init__(task=task, encoder=encoder, head=head)

    def forward(self, batch, tokenizer, compute_loss: bool = False):
        masked_batch = batch.get_masked(
            mlm_probability=self.task.mlm_probability,
            tokenizer=tokenizer,
            do_mask=self.task.do_mask,
        )
        encoder_output = self.encoder.encode(
            input_ids=masked_batch.input_ids,
            segment_ids=masked_batch.segment_ids,
            input_mask=masked_batch.input_mask,
        )
        logits = self.head(unpooled=encoder_output.unpooled)
        if compute_loss:
            loss = compute_mlm_loss(logits=logits, masked_lm_labels=masked_batch.masked_lm_labels)
            return LogitsAndLossOutput(logits=logits, loss=loss, other=encoder_output.other)
        else:
            return LogitsOutput(logits=logits, other=encoder_output.other)


@JiantTaskModelFactory.register(TaskTypes.EMBEDDING)
class EmbeddingModel(Taskmodel):
    def __init__(self, task, encoder, head: heads.AbstractPoolerHead, **kwargs):
        super().__init__(task=task, encoder=encoder, head=head)
        self.layer = kwargs["layer"]

    def forward(self, batch, tokenizer, compute_loss: bool = False):
        encoder_output = self.encoder.encode(
            input_ids=batch.input_ids, segment_ids=batch.segment_ids, input_mask=batch.input_mask,
        )

        # A tuple of layers of hidden states
        hidden_states = take_one(encoder_output.other)
        layer_hidden_states = hidden_states[self.layer]

        if isinstance(self.head, heads.MeanPoolerHead):
            logits = self.head(unpooled=layer_hidden_states, input_mask=batch.input_mask)
        elif isinstance(self.head, heads.FirstPoolerHead):
            logits = self.head(layer_hidden_states)
        else:
            raise TypeError(type(self.head))

        # TODO: Abuse of notation - these aren't really logits  (issue #1187)
        if compute_loss:
            # TODO: make this optional?   (issue #1187)
            return LogitsAndLossOutput(
                logits=logits,
                loss=torch.tensor([0.0]),  # This is a horrible hack
                other=encoder_output.other,
            )
        else:
            return LogitsOutput(logits=logits, other=encoder_output.other)


def compute_mlm_loss(logits, masked_lm_labels):
    vocab_size = logits.shape[-1]
    loss_fct = nn.CrossEntropyLoss()
    return loss_fct(logits.view(-1, vocab_size), masked_lm_labels.view(-1))


def compute_qa_loss(logits, start_positions, end_positions):
    # Do we want to keep them as 1 tensor, or multiple?
    # bs x 2 x seq_len x 1

    start_logits, end_logits = logits[:, 0], logits[:, 1]
    # Taken from: RobertaForQuestionAnswering
    # If we are on multi-GPU, split add a dimension
    if len(start_positions.size()) > 1:
        start_positions = start_positions.squeeze(-1)
    if len(end_positions.size()) > 1:
        end_positions = end_positions.squeeze(-1)
    # sometimes the start/end positions are outside our model inputs, we ignore these terms
    ignored_index = start_logits.size(1)
    start_positions.clamp_(0, ignored_index)
    end_positions.clamp_(0, ignored_index)

    loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss) / 2
    return total_loss
