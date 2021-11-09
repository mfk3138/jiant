import sys
sys.path.insert(0, "/content/jiant")
import jiant.scripts.download_data.runscript as downloader
import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.export_model as export_model
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.runscript as main_runscript
import jiant.shared.caching as caching
import jiant.utils.python.io as py_io
import jiant.utils.display as display
import os
import uuid

EXP_DIR = "/data1/mafukun/export/resource"
DEV_DIR = "/home/mafukun/GLUE/jiant/develop"
addition = "limit_vocab"
task_names = ["mnli_linearized_amr"]
hf_pretrained_model_name = "roberta-base"
run_name = f"mnli_linearized_amr_{addition}"
run_id = uuid.uuid4().hex

# Prepare for task: download data, export model, tokenize and cache
# downloader.download_data(task_names, f"{EXP_DIR}/tasks")

export_model.export_model(
    hf_pretrained_model_name_or_path=hf_pretrained_model_name,
    output_base_path=f"{EXP_DIR}/models/{hf_pretrained_model_name}",
    additional_token_path=f"{DEV_DIR}/additions-limit-2.txt"
)

for task_name in task_names:
    tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
        task_config_path=f"{EXP_DIR}/tasks/configs/{task_name}_{addition}_config.json",
        hf_pretrained_model_name_or_path=f"{EXP_DIR}/models/{hf_pretrained_model_name}/tokenizer",
        output_dir=f"{EXP_DIR}/cache/{task_name}_{addition}",
        phases=["train", "val"],
        max_seq_length=512,
    ))
    row = caching.ChunkedFilesDataCache(f"{EXP_DIR}/cache/{task_name}_{addition}/train").load_chunk(0)[0]["data_row"]
    print(row.input_ids)
    print(row.tokens)

jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
    task_config_path_dict={task_name: f"{EXP_DIR}/tasks/configs/{task_name}_{addition}_config.json"
                           for task_name in task_names},
    task_cache_path_dict={task_name: f"{EXP_DIR}/cache/{task_name}_{addition}" for task_name in task_names},
    train_task_name_list=task_names,
    val_task_name_list=task_names,
    train_batch_size=16,
    eval_batch_size=16,
    epochs=3,
    num_gpus=1,
).create_config()
os.makedirs(f"{EXP_DIR}/run_configs/", exist_ok=True)
py_io.write_json(jiant_run_config, f"{EXP_DIR}/run_configs/{run_name}/{run_id}_run_config.json")
display.show_json(jiant_run_config)

run_args = main_runscript.RunConfiguration(
    jiant_task_container_config_path=f"{EXP_DIR}/run_configs/{run_name}/{run_id}_run_config.json",
    output_dir=f"{EXP_DIR}/runs/{run_name}/{run_id}",
    hf_pretrained_model_name_or_path=hf_pretrained_model_name,
    model_path=f"{EXP_DIR}/models/{hf_pretrained_model_name}/model/model.p",
    model_config_path=f"{EXP_DIR}/models/{hf_pretrained_model_name}/model/config.json",
    learning_rate=1e-5,
    eval_every_steps=1000,
    do_train=True,
    do_val=True,
    do_save=True,
    force_overwrite=True,
)
main_runscript.run_loop(run_args)

