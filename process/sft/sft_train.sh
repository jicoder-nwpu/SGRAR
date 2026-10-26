

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m verl-main.verl.trainer.fsdp_sft_trainer \
    data.train_files=./utility_train_ordinal.parquet \
    data.val_files=./utility_train_ordinal.parquet \
    data.prompt_key=prompt \
    data.response_key=label_str \
    data.max_length=2048 \
    data.micro_batch_size_per_gpu=4 \
	optim.lr=1e-4 \
    model.partial_pretrain=/user_home/jihongru/jihongru/empathetic/ACL-Trans/deepseek-llm-7b-chat \
    trainer.project_name=empatheticdialogue \
    trainer.experiment_name=fanyuyintest \
    trainer.total_epochs=1 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=143 \
    trainer.logger='["console","swanlab"]' \
    # model.lora_rank=32 \
    # model.lora_alpha=64 \
    # model.target_modules=all-linear
    # model.strategy=fsdp \

chmod +x ./genarate.sh
./genarate.sh

