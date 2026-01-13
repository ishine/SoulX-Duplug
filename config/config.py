from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class ASRConfig:
    # infer config
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    sample_rate: int = 16000
    max_wait_num: int = 10
    developer_mode: bool = False

    input: dict = field(
        default_factory=lambda: {
            "chunk_size": 16000,
            "stride": 8000,
            "sample_rate": 16000,
        }
    )
    asr: dict = field(
        default_factory=lambda: {
            "language": "auto",
            "max_chunk_token_length": 256,
            "eos_token_id": 2,
        }
    )
    llm: dict = field(
        default_factory=lambda: {
            "model_path": "Qwen/Qwen2.5-7B-Instruct",
            "temp": 0.8,
            "top_p": 0.95,
            "max_tokens": 3000,
            "tp": 1,
        }
    )

    # model config
    enable_stream: bool = True
    duplex_predict: bool = True
    text_vocab_size: int = 151643
    original_vocab_size: int = 151669
    lm_vocab_size: int = 151936
    tokenizer_vocab_size: int = 203566
    added_audio_token_size: int = 51866
    added_special_token_size: int = 31
    special_token_start: int = 151643
    added_token_start: int = 151669
    added_audio_token_start: int = 151700
    total_vocab_size: int = field(init=False)
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int = 151643
    audio_pad_token_id: int = 151673
    asr_eos_token_id: int = 151674
    asr_bos_token_id: int = 151675

    duplex_action_start_id: int = 151676
    duplex_action_end_id: int = 151679
    duplex_speak_token_id: int = 151676
    duplex_interrupted_token_id: int = 151679
    duplex_interrupt_token_id: int = 151678
    duplex_backchannel_token_id: int = 151677
    duplex_idle_token_id: int = 151680
    duplex_nonidle_token_id: int = 151681

    glm_tokenizer_path: str = "pretrain_models/glm-4-voice-tokenizer"
    model_name: str = "pretrain_models/Qwen3-1.7B-expand_vocab_v1"
    audio_embed_dim: int = 1280
    llm_dim: int = 2048

    continue_in: bool = False
    extract_token_batch: int = 256

    enable_asr_first: bool = False
    enable_audio_mask: bool = False
    enable_new_annotation: bool = False
    enable_cascade_asr: bool = False

    # lora
    enable_lora: bool = False
    lora_task_type: str = "CAUSAL_LM"
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # projector
    enable_projector: bool = True
    freeze_projector: bool = False
    embed_only: bool = False

    # dataset
    batch_size: int = 64
    num_workers: int = 12
    split_size: float = 0.05
    max_token_length: int = 1500
    avg_token_length: int = 150
    train_data_path: str = "data/asr/final/parquet"

    # dynamic batch size
    enable_dynamic_batch: bool = False
    max_batch_size: int = 512
    max_token_per_batch: int = 8192
    stream_dataset_buffer_size: int = 5000

    # testset
    test_data_name: str = ""
    test_lang: str = ""
    test_result_dir: str = ""
    test_result_file: str = ""
    test_data_path: str = "data/asr/final/asr-test"
    asr_repetition_penalty: float = 1.0
    num_beam: int = 3
    punctuation: bool = False
    max_chunk_token_length: int = 50
    input_wav_path: str = ""

    # optimizer
    linear_lr: bool = False
    learning_rate: float = 1e-4
    min_lr: float = 1e-6
    warmup_steps: int = 5000
    anneal_steps: int = 400000
    anneal_rate: float = 0.5
    weight_decay: float = 1e-2
    betas: List[float] = field(init=False)
    eps: float = 1e-8
    asr_eos_loss_rate: float = 0.5
    asr_loss_rate: float = 0.4
    state_loss_rate: float = 0.35
    action_loss_rate: float = 0.25
    idle_loss_rate: float = 0.5

    # tricks
    # EMA
    enable_ema: bool = False
    ema_dacay: float = 0.9
    ema_every_n_steps: int = 1
    ema_start_step: int = 0

    # train adapter first
    adapter_first: bool = False
    unfreeze_step: int = 10000

    # trainer
    seed: int = 42
    stage: str = "train"
    total_steps: int = 1000000
    total_epochs: int = 0
    val_check_interval: int = 10000
    log_every_n_steps: int = 10
    accumulate_grad_batches: int = 1
    num_gpu_per_node: int = 8
    num_node: int = 1
    accelerator: str = "gpu"
    strategy: str = "ddp"
    precision: str = "16-mixed"
    sync_batchnorm: bool = True
    ckpt_path: str = ""
    init_ckpt_path: str = ""
    init_ckpt_path_lora: str = ""
    default_root_dir: str = ""
    debug_log_dir: str = "debug_logs/"
    wandb_run_name: str = ""
    wandb_save_dir: str = ""

    def __post_init__(self):
        self.total_vocab_size = (
            self.original_vocab_size
            + self.added_audio_token_size
            + self.added_special_token_size
        )

        self.betas = [0.9, 0.999]
