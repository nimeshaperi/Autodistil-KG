"""
Unsloth-based fine-tuner: configurable for Gemma, Llama, Qwen, etc.
Requires: unsloth, transformers, trl, datasets (install with: pip install "autodistil-kg[finetune]").
"""
import importlib.abc
import importlib.machinery
import importlib.util
import json
import logging
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Preload NVIDIA pip-installed libs (e.g. cusparseLt) so the dynamic linker finds them.
# Setting LD_LIBRARY_PATH at runtime is too late; use ctypes to load them explicitly.
import ctypes
import ctypes.util

def _preload_nvidia_libs():
    """Find and preload NVIDIA shared libs installed via pip."""
    for base in [
        Path(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia"),
        Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia",
    ]:
        if not base.exists():
            continue
        for so_file in sorted(base.glob("*/lib/libcusparseLt.so*")):
            try:
                ctypes.cdll.LoadLibrary(str(so_file))
                return
            except OSError:
                continue

_preload_nvidia_libs()

from .config import UnslothFineTunerConfig, ModelType, CHAT_TEMPLATE_MAP

logger = logging.getLogger(__name__)


def _emit_finetuner_event(event_type: str, **data) -> None:
    """Emit a structured finetuner event via logging extras.

    The API's QueueLogHandler picks up the ``finetuner_event`` extra and
    forwards it to the WebSocket so the UI can display real-time
    finetuning progress.
    """
    logger.info(
        "finetuner_event: %s %s",
        event_type,
        json.dumps({k: v for k, v in data.items() if v is not None}, default=str),
        extra={"finetuner_event": {"type": event_type, **data}},
    )

# Optional imports are done lazily in methods to avoid loading unsloth/trl/datasets
# at import time (unsloth checks for GPU; users may only want pipeline/chatml).
UNSLOTH_AVAILABLE = None
TRL_AVAILABLE = None
DATASETS_AVAILABLE = None
Dataset = None


def _install_unsloth_import_hook() -> None:
    """Install hook so unsloth.models._utils gets PreTrainedConfig in its namespace.

    Unsloth's _utils does 'from transformers import PretrainedConfig' but the
    exec'd config source uses the name PreTrainedConfig; inject the alias so
    exec(config, globals()) does not raise NameError.
    """
    class _UnslothUtilsLoader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            spec = module.__spec__
            with open(spec.origin, "r", encoding="utf-8") as f:
                source = f.read()
            # Ensure both PreTrainedConfig and PretrainedConfig names are available.
            # Newer transformers only exports PreTrainedConfig; unsloth code may
            # reference PretrainedConfig (lowercase t) after a bare except that
            # doesn't create the alias, causing NameError.
            #
            # Strategy: find the try/except import block and replace it with one
            # that guarantees both names exist.
            _config_try = (
                "try:\n"
                "    from transformers import PreTrainedConfig\n"
                "except:\n"
                "    from transformers import PretrainedConfig"
            )
            _config_fix = (
                "try:\n"
                "    from transformers import PreTrainedConfig\n"
                "    PretrainedConfig = PreTrainedConfig\n"
                "except ImportError:\n"
                "    from transformers import PretrainedConfig\n"
                "    PreTrainedConfig = PretrainedConfig"
            )
            if _config_try in source:
                source = source.replace(_config_try, _config_fix, 1)
            else:
                # Fallback: handle plain import lines without try/except
                needle = "from transformers import PretrainedConfig"
                if needle in source and "PreTrainedConfig = PretrainedConfig" not in source:
                    source = source.replace(
                        needle,
                        needle + "\nPreTrainedConfig = PretrainedConfig",
                        1,
                    )
                needle2 = "from transformers import PreTrainedConfig"
                if needle2 in source and "PretrainedConfig = PreTrainedConfig" not in source:
                    source = source.replace(
                        needle2,
                        needle2 + "\nPretrainedConfig = PreTrainedConfig",
                        1,
                    )
            # RopeParameters was removed in transformers 5.x; only inject when present
            if "from transformers.modeling_rope_utils import RopeParameters" not in source:
                try:
                    from transformers.modeling_rope_utils import RopeParameters  # noqa: F401
                    source = "from transformers.modeling_rope_utils import RopeParameters\n" + source
                except ImportError:
                    pass
            # Skip DynamicCache.__getitem__ patch when transformers has no __getitem__ (e.g. transformers 5.x)
            _cache_cond = (
                'if hasattr(transformers.cache_utils, "DynamicCache") and \\\n'
                "    transformers.cache_utils.DynamicCache.__getitem__.__name__"
            )
            _cache_cond_safe = (
                'if hasattr(transformers.cache_utils, "DynamicCache") and '
                'hasattr(transformers.cache_utils.DynamicCache, "__getitem__") and \\\n'
                "    transformers.cache_utils.DynamicCache.__getitem__.__name__"
            )
            if _cache_cond in source and _cache_cond_safe not in source:
                source = source.replace(_cache_cond, _cache_cond_safe, 1)
            code = compile(source, spec.origin, "exec")
            exec(code, module.__dict__)

    class _UnslothUtilsFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "unsloth.models._utils":
                return None
            try:
                import os
                spec = importlib.util.find_spec("unsloth")
                if spec is None or not spec.submodule_search_locations:
                    return None
                base = spec.submodule_search_locations[0]
                origin = os.path.join(base, "models", "_utils.py")
                if not os.path.isfile(origin):
                    return None
                return importlib.machinery.ModuleSpec(
                    fullname,
                    _UnslothUtilsLoader(),
                    origin=origin,
                )
            except Exception:
                return None

    if not any(type(f).__name__ == "_UnslothUtilsFinder" for f in sys.meta_path):
        sys.meta_path.insert(0, _UnslothUtilsFinder())


def _is_python_dev_headers_error(exc: BaseException) -> bool:
    """Check if the exception is due to missing Python dev headers (Python.h)."""
    msg = str(exc).lower()
    if "python.h" in msg or "compilation terminated" in msg:
        return True
    # subprocess.CalledProcessError during unsloth/triton import is almost always
    # GCC failing due to missing Python.h (Triton compiles C extensions at runtime)
    if isinstance(exc, subprocess.CalledProcessError):
        return True
    cause = exc
    while getattr(cause, "__cause__", None):
        cause = cause.__cause__
        cmsg = str(cause).lower()
        if "python.h" in cmsg or "compilation terminated" in cmsg:
            return True
        if isinstance(cause, subprocess.CalledProcessError):
            return True
    return False


def _raise_finetune_setup_error(exc: BaseException) -> None:
    """Raise a clear RuntimeError with setup instructions for finetuning failures."""
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    raise RuntimeError(
        "Finetuning dependencies failed to load. This is often caused by missing "
        "Python development headers (needed by Triton/CUDA extensions).\n\n"
        f"Install them for Python {py_ver}:\n"
        "  Ubuntu/Debian: sudo apt install python3-dev   # or python3.13-dev\n"
        "  Fedora:        sudo dnf install python3-devel\n"
        "  macOS:         xcode-select --install\n\n"
        "Also ensure the finetune extra is installed:\n"
        "  pip install 'autodistil-kg[finetune]'   # or: poetry install -E finetune\n\n"
        f"Original error: {exc}"
    ) from exc


def _check_unsloth():
    global UNSLOTH_AVAILABLE
    if UNSLOTH_AVAILABLE is not None:
        return UNSLOTH_AVAILABLE
    _install_unsloth_import_hook()
    try:
        from unsloth import FastLanguageModel  # noqa: F401
        from unsloth.chat_templates import get_chat_template, train_on_responses_only  # noqa: F401
        UNSLOTH_AVAILABLE = True
    except subprocess.CalledProcessError as e:
        if _is_python_dev_headers_error(e):
            _raise_finetune_setup_error(e)
        UNSLOTH_AVAILABLE = False
    except Exception as e:
        if _is_python_dev_headers_error(e):
            _raise_finetune_setup_error(e)
        UNSLOTH_AVAILABLE = False
    return UNSLOTH_AVAILABLE


def _check_trl():
    global TRL_AVAILABLE
    if TRL_AVAILABLE is not None:
        return TRL_AVAILABLE
    try:
        from trl import SFTTrainer, SFTConfig  # noqa: F401
        TRL_AVAILABLE = True
    except ImportError:
        TRL_AVAILABLE = False
    return TRL_AVAILABLE


def _check_datasets():
    global DATASETS_AVAILABLE, Dataset
    if DATASETS_AVAILABLE is not None:
        return DATASETS_AVAILABLE
    try:
        from datasets import Dataset as _Dataset
        Dataset = _Dataset
        DATASETS_AVAILABLE = True
    except ImportError:
        Dataset = None
        DATASETS_AVAILABLE = False
    return DATASETS_AVAILABLE


# Default LoRA target modules per family (Unsloth typical)
DEFAULT_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Instruction/response tokens for train_on_responses_only (by template)
RESPONSE_ONLY_PARTS = {
    "gemma3": ("<start_of_turn>user\n", "<start_of_turn>model\n"),
    "gemma4": ("<start_of_turn>user\n", "<start_of_turn>model\n"),
    "llama3": ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "chatml": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "mistral": ("[INST] ", "[/INST] "),
    "qwen2.5": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "qwen3": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "qwen3-thinking": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
}


class _FinetunerProgressCallback:
    """HuggingFace TrainerCallback that emits structured finetuner events.

    Note: We avoid inheriting from TrainerCallback at class definition time
    because transformers may not be imported yet. The SFTTrainer duck-types
    the callback interface, so we just need the right method signatures.
    We add a no-op __getattr__ so missing on_* methods don't raise AttributeError.

    Optionally accepts a ``cancel_token`` to support cooperative cancellation.
    """

    def __init__(self, cancel_token=None):
        self._cancel_token = cancel_token

    def __getattr__(self, name: str):
        if name.startswith("on_"):
            return lambda *args, **kwargs: None
        raise AttributeError(name)

    def _check_cancel(self, control):
        """If cancelled, set trainer control to stop training."""
        if self._cancel_token and self._cancel_token.is_cancelled:
            logger.info("Cancellation requested — stopping training")
            control.should_training_stop = True

    def on_train_begin(self, args, state, control, **kwargs):
        total_steps = state.max_steps if state.max_steps > 0 else 0
        _emit_finetuner_event(
            "training_start",
            total_epochs=args.num_train_epochs,
            total_steps=total_steps,
            batch_size=args.per_device_train_batch_size,
            learning_rate=args.learning_rate,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        self._check_cancel(control)
        if logs is None:
            return
        current_epoch = logs.get("epoch")
        _emit_finetuner_event(
            "training_step",
            step=state.global_step,
            total_steps=state.max_steps if state.max_steps > 0 else 0,
            epoch=round(current_epoch, 2) if current_epoch is not None else None,
            total_epochs=args.num_train_epochs,
            loss=round(logs["loss"], 4) if "loss" in logs else None,
            learning_rate=logs.get("learning_rate"),
            grad_norm=round(logs["grad_norm"], 4) if "grad_norm" in logs else None,
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        self._check_cancel(control)
        current_epoch = math.floor(state.epoch) if state.epoch else 0
        _emit_finetuner_event(
            "epoch_done",
            epoch=current_epoch,
            total_epochs=args.num_train_epochs,
            step=state.global_step,
            total_steps=state.max_steps if state.max_steps > 0 else 0,
        )

    def on_train_end(self, args, state, control, **kwargs):
        _emit_finetuner_event(
            "training_end",
            total_steps=state.global_step,
            train_loss=round(state.log_history[-1].get("train_loss", 0), 4) if state.log_history else None,
        )


def _load_jsonl_messages(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line) 
            if "messages" in item:
                data.append(item)
            elif "conversations" in item:
                messages = [{"role": m["role"], "content": m["content"]} for m in item["conversations"]]
                data.append({"messages": messages})
            else:
                data.append(item)
    return data


def _format_prompts_func(tokenizer, examples: Dict[str, List]) -> Dict[str, List]:
    batched_messages = examples["messages"]
    formatted_convos = []
    formatted_texts = []
    for convo in batched_messages:
        ordered = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in convo
        ]
        formatted_convos.append(ordered)
        text = tokenizer.apply_chat_template(
            ordered,
            tokenize=False,
            add_generation_prompt=False,
        )
        formatted_texts.append(text)
    return {"messages": formatted_convos, "text": formatted_texts}


class UnslothFineTuner:
    """Fine-tune a language model with Unsloth (LoRA). Supports Gemma, Llama, Qwen, etc."""

    def __init__(self, config: UnslothFineTunerConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_model_and_tokenizer(self):
        """Load base model and tokenizer with LoRA setup."""
        if not _check_unsloth():
            raise RuntimeError("unsloth is not installed or not available (e.g. requires NVIDIA GPU). Install with: pip install unsloth ...")
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        model_name = self.config.model_name
        token = self.config.hf_token
        logger.info("Loading model: %s (max_seq_length=%d, 4bit=%s)", model_name, self.config.max_seq_length, self.config.load_in_4bit)
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=self.config.max_seq_length,
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
                full_finetuning=self.config.full_finetuning,
                token=token,
            )
        except Exception as e:
            msg = str(e)
            if "not supported yet in" in msg and "transformers" in msg:
                logger.info("Model %s requires a newer transformers — attempting upgrade…", model_name)
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "--upgrade", "transformers"],
                    stdout=subprocess.DEVNULL,
                )
                # Reload transformers modules so the new version is picked up
                for mod_name in sorted(sys.modules, reverse=True):
                    if mod_name == "transformers" or mod_name.startswith("transformers."):
                        del sys.modules[mod_name]
                # Retry after upgrade
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=self.config.max_seq_length,
                    load_in_4bit=self.config.load_in_4bit,
                    load_in_8bit=self.config.load_in_8bit,
                    full_finetuning=self.config.full_finetuning,
                    token=token,
                )
            else:
                raise
        logger.info("Model loaded successfully")
        target_modules = self.config.lora_target_modules or DEFAULT_LORA_TARGETS
        logger.info("Applying LoRA adapter (r=%d, alpha=%d, targets=%s)", self.config.lora_r, self.config.lora_alpha, target_modules)
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config.lora_r,
            target_modules=target_modules,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
            random_state=self.config.random_state,
            use_rslora=self.config.use_rslora,
            loftq_config=None,
        )
        chat_template_name = self.config.get_chat_template_name()
        tokenizer = get_chat_template(tokenizer, chat_template=chat_template_name)
        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    def load_dataset(self, train_path: Optional[str] = None, eval_path: Optional[str] = None):
        """Load train (and optionally eval) dataset from JSONL with 'messages' key."""
        if not _check_datasets():
            raise RuntimeError("datasets is not installed. Install with: pip install datasets")
        train_path = train_path or self.config.train_data_path
        if not train_path or not Path(train_path).exists():
            raise FileNotFoundError(f"Train data not found: {train_path}")
        raw = _load_jsonl_messages(train_path)
        # Ensure each item has "messages" as list of {role, content}
        for item in raw:
            if "messages" not in item:
                raise ValueError("Each JSONL line must have 'messages' (list of {role, content})")
        train_dataset = Dataset.from_list(raw)
        eval_dataset = None
        ep = eval_path or self.config.eval_data_path
        if ep and Path(ep).exists():
            eval_raw = _load_jsonl_messages(ep)
            eval_dataset = Dataset.from_list(eval_raw)
        return train_dataset, eval_dataset

    def prepare_dataset(self, train_dataset, eval_dataset=None):
        """Apply chat template to produce 'text' field."""
        def format_fn(examples):
            return _format_prompts_func(self.tokenizer, examples)

        parsed = train_dataset.map(
            format_fn,
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        eval_parsed = None
        if eval_dataset and len(eval_dataset) > 0:
            eval_parsed = eval_dataset.map(
                format_fn,
                batched=True,
                remove_columns=eval_dataset.column_names,
            )
        return parsed, eval_parsed

    def get_response_only_parts(self):
        """Return (instruction_part, response_part) for train_on_responses_only."""
        if self.config.instruction_part and self.config.response_part:
            return self.config.instruction_part, self.config.response_part
        template_name = self.config.get_chat_template_name()
        return RESPONSE_ONLY_PARTS.get(template_name, ("<|im_start|>user\n", "<|im_start|>assistant\n"))

    def train(
        self,
        train_path: Optional[str] = None,
        eval_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        cancel_token=None,
    ):
        """Load model, prepare data, run training, save to output_dir."""
        import os
        import shutil

        # CRITICAL: Set env vars and clear stale cache BEFORE importing unsloth
        # Unsloth must be imported before trl/transformers/peft to apply patches correctly
        os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

        # Clear stale unsloth compiled cache to avoid torch.compile option incompatibilities
        # (e.g. cuda.cutlass_epilogue_fusion_enabled was removed in newer PyTorch)
        cache_dirs = [
            Path("/tmp/unsloth_compiled_cache"),
            Path.cwd() / "unsloth_compiled_cache",
            Path(__file__).parent.parent.parent.parent.parent.parent / "Autodistil-KG_api" / "unsloth_compiled_cache",
        ]
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                try:
                    shutil.rmtree(cache_dir)
                    logger.info("Cleared stale unsloth cache: %s", cache_dir)
                except Exception as e:
                    logger.warning("Could not clear unsloth cache %s: %s", cache_dir, e)

        # Install import hook before any unsloth imports
        _install_unsloth_import_hook()

        # IMPORTANT: Import unsloth FIRST, before trl/transformers/peft
        # This allows unsloth to apply its patches correctly
        try:
            import unsloth  # noqa: F401 - must be imported first to patch other libraries
        except subprocess.CalledProcessError as e:
            if _is_python_dev_headers_error(e):
                _raise_finetune_setup_error(e)
            raise
        except Exception as e:
            if _is_python_dev_headers_error(e):
                _raise_finetune_setup_error(e)
            raise

        # Now safe to import trl and transformers (unsloth has patched them)
        if not _check_trl():
            raise RuntimeError("trl is not installed. Install with: pip install trl")

        # Transformers 5.0 renamed AutoModelForVision2Seq -> AutoModelForImageTextToText; unsloth still expects the old name
        import transformers as _tf
        if not hasattr(_tf, "AutoModelForVision2Seq"):
            from transformers import AutoModelForImageTextToText
            _tf.AutoModelForVision2Seq = AutoModelForImageTextToText  # type: ignore[attr-defined]

        from trl import SFTTrainer, SFTConfig
        from unsloth.chat_templates import train_on_responses_only

        _emit_finetuner_event("loading_model", model_name=self.config.model_name)
        self.load_model_and_tokenizer()
        _emit_finetuner_event("model_loaded", model_name=self.config.model_name)

        logger.info("Loading dataset from %s", train_path or self.config.train_data_path)
        _emit_finetuner_event("loading_data")
        train_ds, eval_ds = self.load_dataset(train_path, eval_path)
        logger.info("Dataset loaded: %d train samples%s", len(train_ds), f", {len(eval_ds)} eval samples" if eval_ds else "")

        logger.info("Preparing dataset (applying chat template)...")
        _emit_finetuner_event("preparing_data", train_samples=len(train_ds), eval_samples=len(eval_ds) if eval_ds else 0)
        train_parsed, eval_parsed = self.prepare_dataset(train_ds, eval_ds)
        logger.info("Dataset prepared: %d train examples ready", len(train_parsed))
        _emit_finetuner_event("data_ready", train_samples=len(train_parsed), eval_samples=len(eval_parsed) if eval_parsed else 0)

        out = output_dir or self.config.output_dir
        Path(out).mkdir(parents=True, exist_ok=True)

        sft_config = SFTConfig(
            dataset_text_field=self.config.dataset_text_field,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            num_train_epochs=self.config.num_train_epochs,
            max_steps=self.config.max_steps if self.config.max_steps > 0 else -1,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            seed=self.config.seed,
            report_to=self.config.report_to,
            output_dir=out,
        )
        progress_callback = _FinetunerProgressCallback(cancel_token=cancel_token)
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_parsed,
            eval_dataset=eval_parsed,
            args=sft_config,
            callbacks=[progress_callback],
        )
        inst_part, resp_part = self.get_response_only_parts()
        trainer = train_on_responses_only(trainer, instruction_part=inst_part, response_part=resp_part)
        self.trainer = trainer

        logger.info(
            "Training starting: epochs=%s, batch_size=%d, lr=%s, output=%s",
            self.config.num_train_epochs, self.config.per_device_train_batch_size,
            self.config.learning_rate, out,
        )
        trainer.train(resume_from_checkpoint=self.config.resume_from_checkpoint)

        _emit_finetuner_event("saving_model", output_dir=out)
        logger.info("Training complete, saving model...")
        trainer.save_model(out)
        self.tokenizer.save_pretrained(out)
        _emit_finetuner_event("model_saved", output_dir=out)
        logger.info("Model and tokenizer saved to %s", out)
        return out
