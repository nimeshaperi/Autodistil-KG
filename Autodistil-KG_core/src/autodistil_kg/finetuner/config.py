"""
Finetuner configuration: model types, LoRA, SFT, and paths.
"""
from pydantic import BaseModel
from enum import Enum
from typing import Any, Dict, List, Optional


class ModelType(str, Enum):
    """Supported model families for Unsloth finetuning."""
    GEMMA = "gemma"
    GEMMA2 = "gemma2"
    GEMMA3 = "gemma3"
    GEMMA4 = "gemma4"
    LLAMA = "llama"
    LLAMA3 = "llama3"
    MISTRAL = "mistral"
    QWEN = "qwen"
    QWEN2 = "qwen2"
    QWEN2_MOE = "qwen2_moe"
    QWEN3 = "qwen3"
    QWEN3_5 = "qwen3_5"
    PHI = "phi3"
    PHI4 = "phi4"
    ZEPHYR = "zephyr"
    CHATML = "chatml"
    ALPACA = "alpaca"
    VICUNA = "vicuna"


# Map model_type to unsloth get_chat_template name
CHAT_TEMPLATE_MAP = {
    ModelType.GEMMA: "gemma3",
    ModelType.GEMMA2: "gemma3",
    ModelType.GEMMA3: "gemma3",
    ModelType.GEMMA4: "gemma4",
    ModelType.LLAMA: "llama3",
    ModelType.LLAMA3: "llama3",
    ModelType.MISTRAL: "mistral",
    ModelType.QWEN: "qwen2.5",
    ModelType.QWEN2: "qwen2.5",
    ModelType.QWEN2_MOE: "qwen2",
    ModelType.QWEN3: "qwen3-thinking",
    ModelType.QWEN3_5: "qwen3",
    ModelType.PHI: "phi3",
    ModelType.PHI4: "phi4",
    ModelType.ZEPHYR: "zephyr",
    ModelType.CHATML: "chatml",
    ModelType.ALPACA: "alpaca",
    ModelType.VICUNA: "vicuna",
}

ChatTemplateName = str


class UnslothFineTunerConfig(BaseModel):
    """Configuration for Unsloth-based fine-tuning."""

    # Model
    model_name: str  # e.g. "unsloth/gemma-2-2b-it", "unsloth/llama-3.2-3b"
    model_type: Optional[ModelType] = None  # Inferred from model_name if None
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False
    hf_token: Optional[str] = None

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: Optional[List[str]] = None
    use_gradient_checkpointing: str = "unsloth"
    use_rslora: bool = False
    random_state: int = 3407

    # Chat template
    chat_template: Optional[ChatTemplateName] = None

    # Data
    train_data_path: str = ""
    eval_data_path: Optional[str] = None
    dataset_text_field: str = "text"

    # Training (SFTConfig-style)
    output_dir: str = "./output"
    num_train_epochs: int = 1
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 0
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    report_to: str = "none"
    resume_from_checkpoint: bool = False

    # train_on_responses_only
    instruction_part: Optional[str] = None
    response_part: Optional[str] = None

    additional_params: Dict[str, Any] = {}

    def get_chat_template_name(self) -> str:
        if self.chat_template:
            return self.chat_template
        model_type = self.model_type or self._infer_model_type()
        return CHAT_TEMPLATE_MAP.get(model_type, "chatml")

    def _infer_model_type(self) -> ModelType:
        name = (self.model_name or "").lower()
        if "gemma" in name:
            if "gemma-4" in name or "gemma4" in name:
                return ModelType.GEMMA4
            if "gemma-3" in name or "gemma3" in name:
                return ModelType.GEMMA3
            return ModelType.GEMMA2
        if "llama" in name:
            return ModelType.LLAMA3
        if "qwen" in name:
            if "qwen3.5" in name or "qwen3_5" in name or "qwen-3.5" in name:
                return ModelType.QWEN3_5
            if "qwen3" in name or "qwen-3" in name:
                return ModelType.QWEN3
            return ModelType.QWEN2
        if "mistral" in name:
            return ModelType.MISTRAL
        if "phi" in name:
            return ModelType.PHI4 if "4" in name else ModelType.PHI
        return ModelType.CHATML
