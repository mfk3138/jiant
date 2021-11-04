import os

import torch
from transformers import AutoModelForPreTraining, AutoTokenizer

import jiant.utils.python.io as py_io
import jiant.utils.zconf as zconf


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    hf_pretrained_model_name_or_path = zconf.attr(type=str)
    output_base_path = zconf.attr(type=str)


def export_model(
    hf_pretrained_model_name_or_path: str, output_base_path: str, additional_token_path: str = None
):
    """Retrieve model and tokenizer from Transformers and save all necessary data
    Things saved:
    - Model weights
    - Model config JSON (corresponding to corresponding Transformers model Config object)
    - Tokenizer data
    - JSON file pointing to paths for the above
    Args:
        hf_pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                        Can be either:

                            - A string, the `model id` of a pretrained model configuration
                              hosted inside a model repo on okhuggingface.co.
                              Valid model ids can be located at the root-level, like
                              ``bert-base-uncased``, or namespaced under a user
                              or organization name, like ``dbmdz/bert-base-german-cased``.
                            - A path to a `directory` containing a configuration file saved using
                              the :meth:`~transformers.PretrainedConfig.save_pretrained` method,
                              or the
                              :meth:`~transformers.PreTrainedModel.save_pretrained` method,
                              e.g., ``./my_model_directory/``.
                            - A path or url to a saved configuration JSON `file`, e.g.,
                              ``./my_model_directory/configuration.json``.
        output_base_path: Base path to save output to
        additional_token_path: Path of additional tokens
    """
    model = AutoModelForPreTraining.from_pretrained(hf_pretrained_model_name_or_path)

    model_fol_path = os.path.join(output_base_path, "model")
    model_path = os.path.join(model_fol_path, "model.p")
    model_config_path = os.path.join(model_fol_path, "config.json")
    tokenizer_fol_path = os.path.join(output_base_path, "tokenizer")

    os.makedirs(tokenizer_fol_path, exist_ok=True)
    os.makedirs(model_fol_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(hf_pretrained_model_name_or_path, use_fast=False)
    if additional_token_path:
        add_special_token(model, tokenizer, additional_token_path)

    torch.save(model.state_dict(), model_path)
    model_config_dict = model.config.to_dict()
    model_config_dict["additional_token_path"] = additional_token_path
    py_io.write_json(model_config_dict, model_config_path)
    tokenizer.save_pretrained(tokenizer_fol_path)
    config = {
        "hf_pretrained_model_name_or_path": hf_pretrained_model_name_or_path,
        "model_path": model_path,
        "model_config_path": model_config_path,
    }
    py_io.write_json(config, os.path.join(output_base_path, "config.json"))
    py_io.write_json(config, os.path.join(output_base_path, "tokenizer", "config.json"))


def add_special_token(model, tokenizer, token_file_path):
    special_token_dict = {"additional_special_tokens":["<AMR>"]}
    additional_tokens = []
    init = " "
    for line in py_io.read_file_lines(token_file_path):
        token = line.strip().split()[0]
        additional_tokens.append(token)
    for cnt in range(512):
        additional_tokens.append(f"<pointer:{cnt}>")
    additional_tokens = [init + t if t[0] not in ('_', '-') else t for t in additional_tokens]
    tokenizer.add_special_tokens(special_token_dict)
    tokenizer.add_tokens(additional_tokens)
    if model:
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def main():
    args = RunConfiguration.default_run_cli()
    export_model(
        hf_pretrained_model_name_or_path=args.hf_pretrained_model_name_or_path,
        output_base_path=args.output_base_path,
    )


if __name__ == "__main__":
    main()
