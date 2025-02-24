set -x

export HOME=/mnt/2050data/wentao.zhang/verl_vlm
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=4025943f5c98398d235eae04243f882b45bcd591

conda activate verl

# prepare environment
cd $HOME
#pip install -e .
#pip3 install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
#pip3 install --use-pep517 flash-attn --no-build-isolation
#pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

nnodes=1
n_gpus_per_node=8
tensor_model_parallel_size=2
project_name='verl'
experiment_name='verl_Qwen2.5-VL-3B-Instruct_PPO'

model_name=hub/Qwen2.5-VL-3B-Instruct
critic_model_name=hub/Qwen2.5-3B-Instruct
mm_train_path=$HOME/data/processed/multimodal_math/train.parquet
mm_test_path=$HOME/data/processed/multimodal_math/test.parquet
gsm8k_train_path=$HOME/data/processed/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/processed/gsm8k/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

python3 -m verl.trainer.main_ppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.val_batch_size=1312 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=False \
    critic.model.path=$critic_model_name \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=0.0001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 $@
