#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train/calibrate REAP-pruned checkpoints with comparable recovery strategies."""

import argparse
import csv
import gc
import inspect
import json
import math
import os
import random
import re
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


STRATEGIES = {
    "direct_router_logit_matching",
    "only_router_kd",
    "only_router_finetune",
    "router_top8_expert_kd",
    "router_top8_expert_finetune",
    "router_top16_expert_kd",
    "router_top16_expert_finetune",
    "router_top50_expert_kd",
    "router_top50_expert_finetune",
    "router_top128_expert_kd",
    "router_top128_expert_finetune",
    "full_kd",
    "full_finetune",
}

SIDECAR_FILES = (
    "retained_expert_indices.json",
    "pruned_expert_indices.json",
    "reap_args.yaml",
    "generation_config.json",
    "chat_template.jinja",
)


class TextLineDataset(Dataset):
    def __init__(self, path: str, max_samples: Optional[int] = None):
        self.samples: List[str] = []
        if not os.path.exists(path):
            raise FileNotFoundError(f"Calibration file not found: {path}")

        if path.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if text:
                        self.samples.append(text)
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = obj.get("text") or obj.get("input")
                    if text:
                        self.samples.append(text)
        else:
            raise ValueError("Unsupported dataset file type. Use .txt or .json")

        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise ValueError("No non-empty samples were loaded from the dataset.")

        print(f"[Dataset] Loaded {len(self.samples)} samples from {path}", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> str:
        return self.samples[idx]


@dataclass
class KDCollator:
    tokenizer: AutoTokenizer
    max_length: int

    def __call__(self, batch: List[str]) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }


@dataclass
class CausalLMCollator:
    tokenizer: AutoTokenizer
    max_length: int

    def __call__(self, batch: List[str]) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }


class GPUMonitor:
    def __init__(
        self,
        gpu_ids: Sequence[str],
        csv_path: str,
        metrics_path: str,
        interval_seconds: float,
        ratio: str,
        strategy: str,
    ):
        self.gpu_ids = [str(gpu_id).strip() for gpu_id in gpu_ids if str(gpu_id).strip()]
        self.csv_path = Path(csv_path)
        self.process_csv_path = self.csv_path.with_name(f"{self.csv_path.stem}.processes.csv")
        self.metrics_path = Path(metrics_path)
        self.interval_seconds = float(interval_seconds)
        self.ratio = ratio
        self.strategy = strategy
        self.root_pid = os.getpid()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_wall = 0.0
        self._end_wall = 0.0
        self._start_perf = 0.0
        self._end_perf = 0.0
        self._active = False
        self._current_segment = ""
        self._segments: List[Dict[str, float]] = []
        self.errors: List[str] = []
        self.process_errors: List[str] = []

    def __enter__(self):
        self.start("training_loop")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        self.write_metrics()
        return False

    @contextmanager
    def segment(self, segment_name: str):
        self.start(segment_name)
        try:
            yield self
        finally:
            self.stop()

    def start(self, segment_name: str = "training_loop") -> None:
        if self._active:
            raise RuntimeError("GPU monitor is already active.")
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._segments:
            with open(self.csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "segment",
                        "relative_seconds",
                        "timestamp",
                        "gpu_index",
                        "power_watt",
                        "utilization_percent",
                        "memory_used_mib",
                    ]
                )
            with open(self.process_csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "segment",
                        "relative_seconds",
                        "sample_epoch",
                        "gpu_uuid",
                        "pid",
                        "process_name",
                        "used_memory_mib",
                        "is_own_process",
                    ]
                )
        self._stop = threading.Event()
        self._current_segment = segment_name
        self._start_wall = time.time()
        self._start_perf = time.perf_counter()
        self._active = True
        self._thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)
        self._thread.start()
        print(
            f"[GPU] Monitoring started for GPUs={','.join(self.gpu_ids) or 'auto'} "
            f"segment={segment_name} interval={self.interval_seconds}s csv={self.csv_path}",
            flush=True,
        )

    def stop(self) -> None:
        if not self._active:
            return
        self._end_perf = time.perf_counter()
        self._end_wall = time.time()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_seconds * 3.0))
        duration_seconds = max(self._end_perf - self._start_perf, 0.0)
        self._segments.append(
            {
                "segment": self._current_segment,
                "start_epoch": self._start_wall,
                "end_epoch": self._end_wall,
                "duration_seconds": duration_seconds,
            }
        )
        self._thread = None
        self._active = False
        print(f"[GPU] Monitoring stopped for segment={self._current_segment}.", flush=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_seconds)

    def _sample_once(self) -> None:
        query = "timestamp,index,power.draw,utilization.gpu,memory.used"
        cmd = [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
        if self.gpu_ids:
            cmd.insert(1, f"--id={','.join(self.gpu_ids)}")

        result = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip() or "nvidia-smi failed"
            self.errors.append(message)
            return

        rel = time.perf_counter() - self._start_perf
        rows = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 5:
                continue
            timestamp, gpu_index, power_watt, util_percent, memory_used_mib = parts[:5]
            try:
                power_value = float(power_watt)
                util_value = float(util_percent)
                memory_value = float(memory_used_mib)
            except ValueError:
                continue
            rows.append(
                [
                    self._current_segment,
                    f"{rel:.6f}",
                    timestamp,
                    gpu_index,
                    power_value,
                    util_value,
                    memory_value,
                ]
            )

        if rows:
            with open(self.csv_path, "a", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows(rows)
        self._sample_processes_once(rel)

    def _current_process_tree_pids(self) -> Set[int]:
        root_pid = self.root_pid
        parent_by_pid: Dict[int, int] = {}
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                stat_path = os.path.join("/proc", entry, "stat")
                try:
                    with open(stat_path, "r", encoding="utf-8") as handle:
                        stat = handle.read()
                    tail = stat.rsplit(")", 1)[1].strip().split()
                    if len(tail) >= 2:
                        parent_by_pid[int(entry)] = int(tail[1])
                except Exception:
                    continue
        except Exception as exc:
            self.process_errors.append(f"process tree scan failed: {exc}")
            return {root_pid}

        owned = {root_pid}
        changed = True
        while changed:
            changed = False
            for pid, ppid in parent_by_pid.items():
                if ppid in owned and pid not in owned:
                    owned.add(pid)
                    changed = True
        return owned

    def _sample_processes_once(self, rel: float) -> None:
        query = "gpu_uuid,pid,process_name,used_memory"
        cmd = [
            "nvidia-smi",
            f"--query-compute-apps={query}",
            "--format=csv,noheader,nounits",
        ]
        if self.gpu_ids:
            cmd.insert(1, f"--id={','.join(self.gpu_ids)}")

        result = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip() or "nvidia-smi compute-app query failed"
            self.process_errors.append(message)
            return

        own_pids = self._current_process_tree_pids()
        sample_epoch = f"{time.time():.6f}"
        rows = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue
            gpu_uuid, pid_text, process_name, used_memory_mib = parts[:4]
            try:
                pid = int(pid_text)
                memory_value = float(used_memory_mib)
            except ValueError:
                continue
            rows.append(
                [
                    self._current_segment,
                    f"{rel:.6f}",
                    sample_epoch,
                    gpu_uuid,
                    pid,
                    process_name,
                    memory_value,
                    1 if pid in own_pids else 0,
                ]
            )

        if not rows:
            rows.append([self._current_segment, f"{rel:.6f}", sample_epoch, "", "", "", 0.0, ""])
        with open(self.process_csv_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows(rows)

    def write_metrics(self) -> None:
        duration_seconds = sum(float(segment["duration_seconds"]) for segment in self._segments)
        samples_by_gpu: Dict[str, List[Tuple[float, float, float]]] = {}
        total_memory_by_sample: Dict[Tuple[str, str], float] = {}
        process_sample_keys: Set[Tuple[str, str]] = set()
        own_memory_by_sample: Dict[Tuple[str, str], float] = {}
        foreign_memory_by_sample: Dict[Tuple[str, str], float] = {}
        foreign_processes: Dict[int, str] = {}
        process_rows = 0
        if self.csv_path.exists():
            with open(self.csv_path, "r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    gpu_index = row.get("gpu_index", "")
                    try:
                        power = float(row.get("power_watt", "nan"))
                        util = float(row.get("utilization_percent", "nan"))
                        memory_mib = float(row.get("memory_used_mib", "nan"))
                    except ValueError:
                        continue
                    if math.isnan(power) or math.isnan(util) or math.isnan(memory_mib):
                        continue
                    memory_gb = memory_mib / 1024.0
                    samples_by_gpu.setdefault(gpu_index, []).append((power, util, memory_gb))
                    sample_key = (row.get("segment", ""), row.get("relative_seconds", ""))
                    total_memory_by_sample[sample_key] = total_memory_by_sample.get(sample_key, 0.0) + memory_gb
        if self.process_csv_path.exists():
            with open(self.process_csv_path, "r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    sample_key = (row.get("segment", ""), row.get("relative_seconds", ""))
                    process_sample_keys.add(sample_key)
                    pid_text = str(row.get("pid", "")).strip()
                    if not pid_text:
                        continue
                    try:
                        pid = int(pid_text)
                        memory_mib = float(row.get("used_memory_mib", "0") or 0)
                    except ValueError:
                        continue
                    process_rows += 1
                    memory_gb = memory_mib / 1024.0
                    is_own = str(row.get("is_own_process", "")).strip() == "1"
                    if is_own:
                        own_memory_by_sample[sample_key] = own_memory_by_sample.get(sample_key, 0.0) + memory_gb
                    else:
                        foreign_memory_by_sample[sample_key] = foreign_memory_by_sample.get(sample_key, 0.0) + memory_gb
                        foreign_processes[pid] = str(row.get("process_name", "")).strip()

        avg_power_by_gpu = {
            gpu: sum(power for power, _, _ in values) / len(values)
            for gpu, values in sorted(samples_by_gpu.items())
            if values
        }
        avg_util_by_gpu = {
            gpu: sum(util for _, util, _ in values) / len(values)
            for gpu, values in sorted(samples_by_gpu.items())
            if values
        }
        avg_memory_by_gpu = {
            gpu: sum(memory for _, _, memory in values) / len(values)
            for gpu, values in sorted(samples_by_gpu.items())
            if values
        }
        all_utils = [util for values in samples_by_gpu.values() for _, util, _ in values]
        avg_util = sum(all_utils) / len(all_utils) if all_utils else 0.0
        total_memory_samples = list(total_memory_by_sample.values())
        avg_total_gpu_memory_gb = (
            sum(total_memory_samples) / len(total_memory_samples) if total_memory_samples else 0.0
        )
        own_memory_samples = [own_memory_by_sample.get(key, 0.0) for key in process_sample_keys]
        foreign_memory_samples = [foreign_memory_by_sample.get(key, 0.0) for key in process_sample_keys]
        avg_own_process_gpu_memory_gb = (
            sum(own_memory_samples) / len(own_memory_samples) if own_memory_samples else 0.0
        )
        avg_foreign_process_gpu_memory_gb = (
            sum(foreign_memory_samples) / len(foreign_memory_samples) if foreign_memory_samples else 0.0
        )
        peak_foreign_process_gpu_memory_gb = max(foreign_memory_samples) if foreign_memory_samples else 0.0
        contamination_detected = bool(foreign_processes)
        if contamination_detected:
            contamination_status = "invalid"
        elif self.process_errors or not process_sample_keys:
            contamination_status = "unknown"
        else:
            contamination_status = "valid"
        sum_avg_power = sum(avg_power_by_gpu.values())
        duration_hours = duration_seconds / 3600.0
        num_gpus = len(self.gpu_ids) or len(samples_by_gpu)
        segment_names = [str(segment["segment"]) for segment in self._segments]

        metrics = {
            "ratio": self.ratio,
            "strategy": self.strategy,
            "measurement_scope": "+".join(segment_names) if segment_names else "none",
            "measured_segments": self._segments,
            "assigned_gpu_ids": self.gpu_ids,
            "num_gpus": num_gpus,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "training_loop_start_epoch": self._segments[0]["start_epoch"] if self._segments else 0.0,
            "training_loop_end_epoch": self._segments[-1]["end_epoch"] if self._segments else 0.0,
            "training_loop_seconds": duration_seconds,
            "training_loop_hours": duration_hours,
            "measured_loop_hours": duration_hours,
            "num_monitor_samples": sum(len(values) for values in samples_by_gpu.values()),
            "avg_power_watt_by_gpu": avg_power_by_gpu,
            "avg_utilization_percent_by_gpu": avg_util_by_gpu,
            "avg_memory_used_gb_by_gpu": avg_memory_by_gpu,
            "sum_avg_gpu_power_watt": sum_avg_power,
            "avg_gpu_utilization_percent": avg_util,
            "avg_total_gpu_memory_gb": avg_total_gpu_memory_gb,
            "gpu_energy_kwh": sum_avg_power * duration_hours / 1000.0,
            "effective_gpu_hours": num_gpus * duration_hours * avg_util / 100.0,
            "gpu_memory_gb_hours": avg_total_gpu_memory_gb * duration_hours,
            "gpu_log_csv": str(self.csv_path),
            "process_monitor_csv": str(self.process_csv_path),
            "process_tree_root_pid": self.root_pid,
            "process_monitor_samples": len(process_sample_keys),
            "process_monitor_rows": process_rows,
            "avg_own_process_gpu_memory_gb": avg_own_process_gpu_memory_gb,
            "avg_foreign_process_gpu_memory_gb": avg_foreign_process_gpu_memory_gb,
            "peak_foreign_process_gpu_memory_gb": peak_foreign_process_gpu_memory_gb,
            "contamination_status": contamination_status,
            "contamination_detected": contamination_detected,
            "contamination_num_foreign_pids": len(foreign_processes),
            "contamination_foreign_processes": [
                {"pid": pid, "process_name": foreign_processes[pid]}
                for pid in sorted(foreign_processes)
            ],
            "monitor_errors": self.errors[:20],
            "process_monitor_errors": self.process_errors[:20],
        }
        with open(self.metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"[GPU] Metrics saved to: {self.metrics_path}", flush=True)


def parse_gpu_ids(value: Optional[str]) -> List[str]:
    raw = value or os.environ.get("ASSIGNED_GPU_IDS") or os.environ.get("CUDA_VISIBLE_DEVICES") or ""
    if not raw:
        return [str(idx) for idx in range(torch.cuda.device_count())]
    return [part.strip() for part in raw.split(",") if part.strip()]


def build_gpu_monitor(args: argparse.Namespace) -> GPUMonitor:
    return GPUMonitor(
        gpu_ids=parse_gpu_ids(args.assigned_gpu_ids),
        csv_path=args.gpu_log_csv,
        metrics_path=args.gpu_metrics_json,
        interval_seconds=args.gpu_log_interval,
        ratio=args.ratio,
        strategy=args.strategy,
    )


def set_reproducibility(seed: int, deterministic: bool) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    try:
        import numpy as np

        np.random.seed(worker_seed + worker_id)
    except Exception:
        pass


def count_parameters(model: nn.Module, only_trainable: bool = False) -> int:
    if only_trainable:
        return sum(param.numel() for param in model.parameters() if param.requires_grad)
    return sum(param.numel() for param in model.parameters())


def detect_input_device(model: nn.Module, fallback: torch.device) -> torch.device:
    for param in model.parameters():
        return param.device
    return fallback


def source_checkpoint_path(args: argparse.Namespace) -> str:
    return args.student_model_name_or_path or args.model_name_or_path


def build_model_load_kwargs(args: argparse.Namespace, trainable: bool) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.float32 if args.cpu else torch.bfloat16,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    if not args.cpu and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            kwargs["device_map"] = "auto"
            kwargs["max_memory"] = {
                gpu_idx: f"{args.max_memory_per_gpu_gb}GiB" for gpu_idx in range(gpu_count)
            }
    return kwargs


def load_tokenizer(path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(path: str, args: argparse.Namespace, trainable: bool) -> nn.Module:
    print(f"[Init] Loading model: {path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        **build_model_load_kwargs(args, trainable=trainable),
    )
    if args.cpu or torch.cuda.device_count() <= 1:
        device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
        model.to(device)
    if trainable and args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model


def find_sparse_moe_prefixes(model: nn.Module) -> List[str]:
    module_dict = dict(model.named_modules())
    prefixes: List[str] = []
    seen = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module_name.split(".")[-1] != "gate":
            continue
        prefix = module_name.rsplit(".", 1)[0]
        experts = module_dict.get(prefix + ".experts")
        if experts is None:
            continue
        if prefix in seen:
            continue
        prefixes.append(prefix)
        seen.add(prefix)
    if not prefixes:
        raise RuntimeError("No sparse MoE blocks were found.")
    print(f"[MoE] Found {len(prefixes)} sparse MoE block(s).", flush=True)
    return prefixes


def mark_all_frozen(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def enable_router_training(model: nn.Module) -> List[str]:
    router_names: List[str] = []
    for name, param in model.named_parameters():
        if name.endswith(".gate.weight") and ".experts." not in name:
            param.requires_grad = True
            router_names.append(name)
    if not router_names:
        raise RuntimeError("No router gate parameters were found.")
    print(f"[Router] Enabled {len(router_names)} router tensor(s).", flush=True)
    return router_names


def get_router_collection_forward_kwargs(model: nn.Module) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "use_cache": False,
        "output_router_logits": True,
    }
    try:
        forward_sig = inspect.signature(model.forward)
        if "logits_to_keep" in forward_sig.parameters:
            kwargs["logits_to_keep"] = 1
    except Exception:
        pass
    return kwargs


def get_minimal_forward_kwargs(model: nn.Module) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "use_cache": False,
    }
    try:
        forward_sig = inspect.signature(model.forward)
        if "logits_to_keep" in forward_sig.parameters:
            kwargs["logits_to_keep"] = 1
    except Exception:
        pass
    return kwargs


def get_router_logits(outputs, model_label: str) -> Tuple[torch.Tensor, ...]:
    router_logits = getattr(outputs, "router_logits", None)
    if router_logits is None:
        raise RuntimeError(
            f"{model_label} did not return router_logits with output_router_logits=True."
        )
    if isinstance(router_logits, torch.Tensor):
        return (router_logits,)
    if isinstance(router_logits, list):
        return tuple(router_logits)
    if isinstance(router_logits, tuple):
        return router_logits
    raise RuntimeError(f"Unexpected router_logits type: {type(router_logits)}")


class RouterLogitHookCollector:
    """Collect sparse MoE gate outputs for models that do not return router_logits."""

    def __init__(
        self,
        module_dict: Dict[str, nn.Module],
        prefixes: Sequence[str],
        detach: bool = True,
    ):
        self.prefixes = list(prefixes)
        self.values: Dict[str, torch.Tensor] = {}
        self.handles = []
        self.detach = detach
        for prefix in self.prefixes:
            gate_name = prefix + ".gate"
            gate_module = module_dict.get(gate_name)
            if gate_module is None:
                raise RuntimeError(f"Router gate module not found for hook collection: {gate_name}")
            self.handles.append(gate_module.register_forward_hook(self._make_hook(prefix)))

    def _make_hook(self, prefix: str):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            self.values[prefix] = output.detach() if self.detach else output

        return hook

    def clear(self) -> None:
        self.values.clear()

    def collect(self) -> Tuple[torch.Tensor, ...]:
        missing = [prefix for prefix in self.prefixes if prefix not in self.values]
        if missing:
            raise RuntimeError(
                "Router gate hook did not observe every sparse MoE block. "
                f"Missing examples: {missing[:5]}"
            )
        return tuple(self.values[prefix] for prefix in self.prefixes)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def parse_layer_index(prefix: str) -> int:
    match = re.search(r"\.layers\.(\d+)\.", prefix)
    return int(match.group(1)) if match else -1


def get_logical_to_physical(module: nn.Module, device: torch.device) -> torch.Tensor:
    mapping = getattr(module, "logical_to_physical", None)
    if mapping is None:
        logical = int(getattr(module, "logical_num_experts", getattr(module, "num_experts", 0)))
        return torch.arange(logical, dtype=torch.long, device=device)
    return mapping.to(device=device, dtype=torch.long)



def load_json_int_map(path: Path) -> Dict[int, List[int]]:
    if not path.exists():
        raise FileNotFoundError(f"Required REAP mapping file is missing: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        int(layer_idx): [int(expert_id) for expert_id in expert_ids]
        for layer_idx, expert_ids in payload.items()
    }


def load_reap_retained_mapping(checkpoint_path: str) -> Dict[int, List[int]]:
    return load_json_int_map(Path(checkpoint_path) / "retained_expert_indices.json")


def load_reap_pruned_mapping(checkpoint_path: str) -> Dict[int, List[int]]:
    return load_json_int_map(Path(checkpoint_path) / "pruned_expert_indices.json")


def save_router_logit_mapping_audit(
    args: argparse.Namespace,
    student: nn.Module,
    student_prefixes: Sequence[str],
) -> Dict[int, List[int]]:
    retained_by_layer = load_reap_retained_mapping(source_checkpoint_path(args))
    pruned_by_layer = load_reap_pruned_mapping(source_checkpoint_path(args))
    module_dict = dict(student.named_modules())
    layers: List[Dict[str, object]] = []
    errors: List[str] = []
    original_num_experts = 128

    for prefix in student_prefixes:
        layer_idx = parse_layer_index(prefix)
        module = module_dict[prefix]
        retained = retained_by_layer.get(layer_idx)
        pruned = pruned_by_layer.get(layer_idx)
        expert_count = len(getattr(module, "experts", []))
        gate = getattr(module, "gate", None)
        gate_out_features = int(getattr(gate, "out_features", -1)) if gate is not None else -1
        moe_num_experts = int(getattr(module, "num_experts", expert_count))

        if retained is None:
            errors.append(f"layer {layer_idx}: missing retained_expert_indices entry")
            retained = []
        if pruned is None:
            errors.append(f"layer {layer_idx}: missing pruned_expert_indices entry")
            pruned = []

        retained_set = set(retained)
        pruned_set = set(pruned)
        if len(retained_set) != len(retained):
            errors.append(f"layer {layer_idx}: retained_expert_indices has duplicates")
        if len(pruned_set) != len(pruned):
            errors.append(f"layer {layer_idx}: pruned_expert_indices has duplicates")
        if retained_set & pruned_set:
            errors.append(f"layer {layer_idx}: retained/pruned expert ids overlap")
        if len(retained_set | pruned_set) != original_num_experts:
            errors.append(f"layer {layer_idx}: retained+pruned does not cover {original_num_experts} original experts")
        invalid = [idx for idx in retained + pruned if idx < 0 or idx >= original_num_experts]
        if invalid:
            errors.append(f"layer {layer_idx}: invalid original expert ids: {invalid[:10]}")
        if expert_count != len(retained):
            errors.append(f"layer {layer_idx}: module expert_count {expert_count} != retained length {len(retained)}")
        if gate_out_features != len(retained):
            errors.append(f"layer {layer_idx}: gate_out_features {gate_out_features} != retained length {len(retained)}")
        if moe_num_experts != len(retained):
            errors.append(f"layer {layer_idx}: moe num_experts {moe_num_experts} != retained length {len(retained)}")

        layers.append(
            {
                "layer_index": layer_idx,
                "moe_prefix": prefix,
                "retained_expert_indices": retained,
                "pruned_expert_indices": pruned,
                "student_expert_count": expert_count,
                "student_gate_out_features": gate_out_features,
                "student_moe_num_experts": moe_num_experts,
                "student_expert_id_to_original_expert_id": {
                    str(student_id): int(original_id)
                    for student_id, original_id in enumerate(retained)
                },
            }
        )

    audit = {
        "ratio": args.ratio,
        "strategy": args.strategy,
        "student_checkpoint": source_checkpoint_path(args),
        "retained_expert_indices_json": str(Path(source_checkpoint_path(args)) / "retained_expert_indices.json"),
        "pruned_expert_indices_json": str(Path(source_checkpoint_path(args)) / "pruned_expert_indices.json"),
        "original_num_experts": original_num_experts,
        "num_layers": len(layers),
        "ok": not errors,
        "errors": errors,
        "layers": layers,
    }
    save_json(os.path.join(args.output_dir, "router_logit_matching_mapping_audit.json"), audit)
    if errors:
        raise RuntimeError("REAP expert mapping audit failed:\n" + "\n".join(errors[:20]))
    return retained_by_layer


def align_teacher_router_logits_for_student(
    teacher_layer_logits: torch.Tensor,
    student_layer_logits: torch.Tensor,
    retained_original_expert_ids: Sequence[int],
    layer_idx: int,
) -> torch.Tensor:
    teacher_dim = int(teacher_layer_logits.shape[-1])
    student_dim = int(student_layer_logits.shape[-1])
    teacher_layer_logits = teacher_layer_logits.float()

    if teacher_dim == student_dim:
        return teacher_layer_logits.to(device=student_layer_logits.device)

    retained = torch.tensor(
        [int(expert_id) for expert_id in retained_original_expert_ids],
        dtype=torch.long,
        device=teacher_layer_logits.device,
    )
    if retained.numel() != student_dim:
        raise ValueError(
            f"Layer {layer_idx} retained mapping length {retained.numel()} != student router dim {student_dim}."
        )
    if retained.numel() == 0:
        raise ValueError(f"Layer {layer_idx} retained mapping is empty.")
    if int(retained.min().item()) < 0 or int(retained.max().item()) >= teacher_dim:
        raise ValueError(
            f"Layer {layer_idx} retained original ids exceed teacher router dim {teacher_dim}: "
            f"min={int(retained.min().item())}, max={int(retained.max().item())}."
        )

    aligned = torch.index_select(teacher_layer_logits, dim=-1, index=retained)
    return aligned.to(device=student_layer_logits.device)


def router_logit_matching_loss(
    teacher_router_logits: Sequence[torch.Tensor],
    student_router_logits: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    student_moe_prefixes: Sequence[str],
    retained_expert_indices_by_layer: Dict[int, List[int]],
    temperature: float,
) -> torch.Tensor:
    if len(teacher_router_logits) != len(student_router_logits):
        raise ValueError(
            "Teacher/student router layer counts do not match: "
            f"{len(teacher_router_logits)} vs {len(student_router_logits)}"
        )

    flat_mask = attention_mask.reshape(-1).float()
    ref_device = student_router_logits[0].device
    total_loss = None

    for layer_idx, (teacher_layer_logits, student_layer_logits) in enumerate(
        zip(teacher_router_logits, student_router_logits)
    ):
        student_layer_logits = student_layer_logits.float()
        prefix = student_moe_prefixes[layer_idx]
        model_layer_idx = parse_layer_index(prefix)
        retained_original_ids = retained_expert_indices_by_layer.get(model_layer_idx)
        if retained_original_ids is None:
            raise ValueError(f"Missing retained expert mapping for layer {model_layer_idx}.")

        aligned_teacher_logits = align_teacher_router_logits_for_student(
            teacher_layer_logits=teacher_layer_logits,
            student_layer_logits=student_layer_logits,
            retained_original_expert_ids=retained_original_ids,
            layer_idx=model_layer_idx,
        )
        if tuple(aligned_teacher_logits.shape) != tuple(student_layer_logits.shape):
            raise ValueError(
                f"Aligned router logit shape mismatch at layer {model_layer_idx}: "
                f"teacher={tuple(aligned_teacher_logits.shape)}, "
                f"student={tuple(student_layer_logits.shape)}"
            )

        if temperature != 1.0:
            aligned_teacher_logits = aligned_teacher_logits / temperature
            student_layer_logits = student_layer_logits / temperature

        tokenwise_mse = (student_layer_logits - aligned_teacher_logits).pow(2).mean(dim=-1).reshape(-1)
        if tokenwise_mse.numel() != flat_mask.numel():
            raise ValueError(
                f"Router token count mismatch at layer {model_layer_idx}: "
                f"loss_tokens={tokenwise_mse.numel()} mask_tokens={flat_mask.numel()}"
            )
        layer_mask = flat_mask.to(student_layer_logits.device)
        layer_loss = (tokenwise_mse * layer_mask).sum() / (layer_mask.sum() + 1e-8)
        total_loss = layer_loss.to(ref_device) if total_loss is None else total_loss + layer_loss.to(ref_device)

    return total_loss / max(1, len(student_router_logits))

def select_top_physical_experts_per_layer(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    experts_per_layer: int,
    selection_top_k: Optional[int],
    logging_steps: int,
) -> Tuple[Dict[str, List[int]], List[Dict[str, object]]]:
    if experts_per_layer <= 0:
        raise ValueError("--experts_per_layer must be positive.")

    prefixes = find_sparse_moe_prefixes(model)
    module_dict = dict(model.named_modules())
    first_module = module_dict[prefixes[0]]
    physical_num_experts = int(
        getattr(first_module, "physical_num_experts", len(getattr(first_module, "experts")))
    )
    logical_num_experts = int(getattr(first_module, "logical_num_experts", physical_num_experts))
    router_top_k = selection_top_k or int(getattr(model.config, "num_experts_per_tok", 1))
    router_top_k = max(1, min(router_top_k, logical_num_experts))

    counts = [torch.zeros(physical_num_experts, dtype=torch.long) for _ in prefixes]
    total_valid_tokens = 0
    start_time = time.time()
    forward_kwargs = get_minimal_forward_kwargs(model)
    hook_collector = RouterLogitHookCollector(module_dict, prefixes)

    print(
        f"[Select] Selecting top physical experts: experts_per_layer={experts_per_layer}, "
        f"router_top_k={router_top_k} via gate hooks",
        flush=True,
    )

    model.eval()
    try:
        for step, batch in enumerate(dataloader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            flat_mask_cpu = attention_mask.reshape(-1).bool().cpu()
            total_valid_tokens += int(flat_mask_cpu.sum().item())

            hook_collector.clear()
            with torch.no_grad():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **forward_kwargs,
                )

            router_logits = hook_collector.collect()
            if len(router_logits) != len(prefixes):
                raise RuntimeError(
                    f"Router logits count mismatch: {len(router_logits)} logits vs {len(prefixes)} MoE blocks."
                )

            for layer_idx, layer_logits in enumerate(router_logits):
                valid_mask = flat_mask_cpu.to(layer_logits.device)
                valid_logits = layer_logits[valid_mask]
                if valid_logits.numel() == 0:
                    continue
                logical_ids = torch.topk(valid_logits, k=router_top_k, dim=-1).indices
                mapping = get_logical_to_physical(module_dict[prefixes[layer_idx]], logical_ids.device)
                physical_ids = mapping[logical_ids]
                layer_counts = torch.bincount(
                    physical_ids.reshape(-1).cpu(),
                    minlength=counts[layer_idx].numel(),
                )
                counts[layer_idx] += layer_counts

            if logging_steps > 0 and step % logging_steps == 0:
                elapsed = time.time() - start_time
                print(
                    f"[Select] step={step}/{len(dataloader)} valid_tokens={total_valid_tokens:,} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
    finally:
        hook_collector.close()

    selected_by_prefix: Dict[str, List[int]] = {}
    selection_summary: List[Dict[str, object]] = []
    for prefix, layer_counts in zip(prefixes, counts):
        k = min(experts_per_layer, layer_counts.numel())
        selected = torch.topk(layer_counts, k=k).indices.tolist()
        selected_by_prefix[prefix] = selected
        entry = {
            "layer_index": parse_layer_index(prefix),
            "moe_prefix": prefix,
            "selected_physical_experts": selected,
            "selection_counts": {
                str(expert_id): int(layer_counts[expert_id].item()) for expert_id in selected
            },
        }
        selection_summary.append(entry)
        print(
            f"[Select] layer={entry['layer_index']:02d} prefix={prefix} "
            f"selected_physical_experts={selected}",
            flush=True,
        )
    model.train()
    return selected_by_prefix, selection_summary


def enable_selected_expert_training(
    model: nn.Module,
    selected_experts_by_prefix: Dict[str, List[int]],
) -> List[str]:
    module_dict = dict(model.named_modules())
    selected_prefixes = {
        f"{prefix}.experts.{expert_id}"
        for prefix, expert_ids in selected_experts_by_prefix.items()
        for expert_id in expert_ids
    }
    for module_prefix in selected_prefixes:
        if module_prefix not in module_dict:
            raise RuntimeError(f"Selected expert module not found: {module_prefix}")

    trainable_names: List[str] = []
    for name, param in model.named_parameters():
        if any(name.startswith(prefix + ".") for prefix in selected_prefixes):
            param.requires_grad = True
            trainable_names.append(name)
    if not trainable_names:
        raise RuntimeError("No selected expert parameters were enabled.")
    print(
        f"[Experts] Enabled full training for {len(selected_prefixes)} expert module(s), "
        f"{len(trainable_names)} parameter tensor(s).",
        flush=True,
    )
    return trainable_names


def enable_all_expert_training(
    model: nn.Module,
) -> Tuple[Dict[str, List[int]], List[Dict[str, object]], List[str]]:
    prefixes = find_sparse_moe_prefixes(model)
    module_dict = dict(model.named_modules())
    selected_experts_by_prefix: Dict[str, List[int]] = {}
    selected_summary: List[Dict[str, object]] = []
    selected_prefixes = set()

    for prefix in prefixes:
        experts = module_dict.get(prefix + ".experts")
        if experts is None:
            raise RuntimeError(f"Expert module list not found: {prefix}.experts")
        expert_ids = list(range(len(experts)))
        if not expert_ids:
            raise RuntimeError(f"No expert modules found under: {prefix}.experts")
        selected_experts_by_prefix[prefix] = expert_ids
        selected_prefixes.update(f"{prefix}.experts.{expert_id}" for expert_id in expert_ids)
        selected_summary.append(
            {
                "layer_index": parse_layer_index(prefix),
                "moe_prefix": prefix,
                "selection_method": "all_student_experts_no_selection",
                "selected_physical_experts": expert_ids,
                "selection_counts": {},
            }
        )

    missing_prefixes = sorted(prefix for prefix in selected_prefixes if prefix not in module_dict)
    if missing_prefixes:
        raise RuntimeError(
            "All-expert strategy expected expert modules that were not found. "
            f"Missing examples: {missing_prefixes[:10]}"
        )

    trainable_names: List[str] = []
    for name, param in model.named_parameters():
        if any(name.startswith(prefix + ".") for prefix in selected_prefixes):
            param.requires_grad = True
            trainable_names.append(name)
    if not trainable_names:
        raise RuntimeError("No expert parameters were enabled for all-expert training.")

    print(
        f"[Experts] Enabled all student experts: {len(selected_prefixes)} expert module(s), "
        f"{len(trainable_names)} parameter tensor(s).",
        flush=True,
    )
    return selected_experts_by_prefix, selected_summary, trainable_names


def summarize_trainable_parameters(
    model: nn.Module,
    allowed_selected_experts: Optional[Dict[str, List[int]]] = None,
) -> Tuple[List[nn.Parameter], List[str]]:
    trainable_params: List[nn.Parameter] = []
    trainable_names: List[str] = []
    unexpected: List[str] = []
    selected_prefixes = set()
    if allowed_selected_experts:
        selected_prefixes = {
            f"{prefix}.experts.{expert_id}"
            for prefix, expert_ids in allowed_selected_experts.items()
            for expert_id in expert_ids
        }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        trainable_params.append(param)
        trainable_names.append(name)
        is_router = name.endswith(".gate.weight") and ".experts." not in name
        is_selected_expert = any(name.startswith(prefix + ".") for prefix in selected_prefixes)
        if allowed_selected_experts is not None and not (is_router or is_selected_expert):
            unexpected.append(name)

    if unexpected:
        raise RuntimeError(
            "Found trainable parameters outside router + selected expert scope:\n"
            + "\n".join(unexpected[:50])
        )
    if not trainable_params:
        raise RuntimeError("No trainable parameters found.")

    trainable_count = sum(param.numel() for param in trainable_params)
    total_count = count_parameters(model, only_trainable=False)
    print(
        f"[Params] Trainable params: {trainable_count:,} / {total_count:,} "
        f"({100.0 * trainable_count / max(1, total_count):.4f}%)",
        flush=True,
    )
    return trainable_params, trainable_names


def kd_loss_token_level(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    temperature = float(temperature)
    teacher_logits = teacher_logits.to(device=student_logits.device, dtype=torch.float32)
    student_logits = student_logits.float()

    teacher_logits = teacher_logits[:, :-1, :].contiguous()
    student_logits = student_logits[:, :-1, :].contiguous()
    mask = attention_mask[:, 1:].contiguous().float().to(student_logits.device)

    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = teacher_log_probs.exp()

    kl = torch.sum(teacher_probs * (teacher_log_probs - student_log_probs), dim=-1)
    kl = kl * mask
    loss = kl.sum() / (mask.sum() + 1e-8)
    return loss * (temperature * temperature)




def save_json(path: str, payload: Dict[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def append_loss(path: str, payload: Dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def prepare_output_dir(output_dir: str, allow_overwrite: bool) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    model_markers = [
        "config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ]
    has_model = any((path / marker).exists() for marker in model_markers)
    if has_model and not allow_overwrite:
        raise FileExistsError(
            f"Output directory already contains a saved model: {output_dir}. "
            "Set ALLOW_OVERWRITE=1 or remove the directory before rerunning."
        )


def save_reap_sidecars(source_dir: str, output_dir: str) -> None:
    if not source_dir or not os.path.isdir(source_dir):
        return
    for filename in SIDECAR_FILES:
        source = Path(source_dir) / filename
        target = Path(output_dir) / filename
        if not source.exists():
            continue
        if source.resolve() == target.resolve():
            continue
        shutil.copy2(source, target)
        print(f"[Save] Copied REAP sidecar file: {filename}", flush=True)


def save_model(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    trainable_param_names: Sequence[str],
    selected_experts: Optional[List[Dict[str, object]]] = None,
) -> None:
    prepare_output_dir(args.output_dir, allow_overwrite=args.allow_overwrite)
    print(f"[Save] Saving model to: {args.output_dir}", flush=True)
    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(args.output_dir)
    save_reap_sidecars(source_checkpoint_path(args), args.output_dir)

    summary = {
        "ratio": args.ratio,
        "strategy": args.strategy,
        "teacher_model_name_or_path": args.teacher_model_name_or_path,
        "student_or_model_name_or_path": source_checkpoint_path(args),
        "calib_dataset_path": args.calib_dataset_path,
        "output_dir": args.output_dir,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "temperature": args.temperature,
        "max_calib_samples": args.max_calib_samples,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "gradient_checkpointing": args.gradient_checkpointing,
        "max_memory_per_gpu_gb": args.max_memory_per_gpu_gb,
        "trainable_params": count_parameters(model, only_trainable=True),
        "total_params": count_parameters(model, only_trainable=False),
        "trainable_param_names": list(trainable_param_names),
        "selected_experts": selected_experts or [],
    }
    save_json(os.path.join(args.output_dir, "training_summary.json"), summary)

    if args.save_trainable_state:
        state = {
            name: param.detach().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        state_path = os.path.join(args.output_dir, "trainable_state_dict.pt")
        torch.save(state, state_path)
        print(f"[Save] Trainable state_dict saved to: {state_path}", flush=True)


def build_dataloader(
    dataset: Dataset,
    collator,
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        drop_last=False,
        pin_memory=torch.cuda.is_available() and not args.cpu,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=generator if shuffle else None,
    )



def train_direct_router_logit_matching(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    student: nn.Module,
    teacher: nn.Module,
    trainable_params: Sequence[nn.Parameter],
    dataset: Dataset,
) -> int:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    input_device = detect_input_device(student, device)
    print(f"[Device] Input tensors will be placed on: {input_device}", flush=True)

    collator = KDCollator(tokenizer=tokenizer, max_length=args.max_length)
    dataloader = build_dataloader(dataset, collator, args, shuffle=True)
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        foreach=False,
    )
    optimizer.zero_grad(set_to_none=True)
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_steps = args.max_train_steps or (args.num_epochs * num_update_steps_per_epoch)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    loss_path = os.path.join(args.output_dir, "loss_history.jsonl")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if os.path.exists(loss_path):
        os.remove(loss_path)

    teacher_prefixes = find_sparse_moe_prefixes(teacher)
    student_prefixes = find_sparse_moe_prefixes(student)
    if len(teacher_prefixes) != len(student_prefixes):
        raise RuntimeError(
            f"Teacher/student MoE block count mismatch: {len(teacher_prefixes)} vs {len(student_prefixes)}"
        )
    retained_expert_indices_by_layer = save_router_logit_mapping_audit(args, student, student_prefixes)

    teacher_module_dict = dict(teacher.named_modules())
    student_module_dict = dict(student.named_modules())
    teacher_hook = RouterLogitHookCollector(teacher_module_dict, teacher_prefixes, detach=True)
    student_hook = RouterLogitHookCollector(student_module_dict, student_prefixes, detach=False)
    teacher_forward_kwargs = get_minimal_forward_kwargs(teacher)
    student_forward_kwargs = get_minimal_forward_kwargs(student)

    print(
        f"[Train] Direct router logit matching epochs={args.num_epochs} "
        f"batches/epoch={len(dataloader)} total_steps={total_steps}",
        flush=True,
    )
    teacher.eval()
    student.train()
    global_step = 0
    start_time = time.time()
    gpu_monitor = build_gpu_monitor(args)

    try:
        with gpu_monitor.segment("training_loop"):
            for epoch in range(args.num_epochs):
                epoch_loss = 0.0
                steps_in_epoch = 0
                for step, batch in enumerate(dataloader):
                    input_ids = batch["input_ids"].to(input_device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(input_device, non_blocking=True)

                    teacher_hook.clear()
                    with torch.no_grad():
                        teacher(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            **teacher_forward_kwargs,
                        )
                        teacher_router_logits = teacher_hook.collect()

                    student_hook.clear()
                    student(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **student_forward_kwargs,
                    )
                    student_router_logits = student_hook.collect()

                    raw_loss = router_logit_matching_loss(
                        teacher_router_logits=teacher_router_logits,
                        student_router_logits=student_router_logits,
                        attention_mask=attention_mask,
                        student_moe_prefixes=student_prefixes,
                        retained_expert_indices_by_layer=retained_expert_indices_by_layer,
                        temperature=args.temperature,
                    )
                    loss = raw_loss / args.gradient_accumulation_steps
                    loss.backward()

                    epoch_loss += raw_loss.detach().float().item()
                    steps_in_epoch += 1
                    should_step = (
                        (step + 1) % args.gradient_accumulation_steps == 0
                        or (step + 1) == len(dataloader)
                    )
                    if not should_step:
                        continue

                    torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm, foreach=False)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    current_loss = raw_loss.detach().float().item()
                    current_lr = lr_scheduler.get_last_lr()[0]
                    append_loss(
                        loss_path,
                        {
                            "event": "step",
                            "time": time.time(),
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "total_steps": total_steps,
                            "loss": current_loss,
                            "lr": current_lr,
                        },
                    )
                    if global_step % args.logging_steps == 0:
                        elapsed = max(time.time() - start_time, 1e-8)
                        steps_per_sec = global_step / elapsed
                        remaining_steps = max(total_steps - global_step, 0)
                        eta_seconds = remaining_steps / max(steps_per_sec, 1e-8)
                        print(
                            f"[Step {global_step}/{total_steps}] router_mse={current_loss:.6f} "
                            f"lr={current_lr:.8f} {steps_per_sec:.3f} step/s "
                            f"ETA {int(eta_seconds // 3600):02d}:"
                            f"{int((eta_seconds % 3600) // 60):02d}:"
                            f"{int(eta_seconds % 60):02d}",
                            flush=True,
                        )

                    if args.max_train_steps is not None and global_step >= args.max_train_steps:
                        break

                avg_epoch_loss = epoch_loss / max(1, steps_in_epoch)
                append_loss(
                    loss_path,
                    {
                        "event": "epoch",
                        "time": time.time(),
                        "epoch": epoch + 1,
                        "avg_loss": avg_epoch_loss,
                        "global_step": global_step,
                    },
                )
                print(f"[Epoch {epoch + 1}/{args.num_epochs}] avg_router_mse={avg_epoch_loss:.6f}", flush=True)

                if args.max_train_steps is not None and global_step >= args.max_train_steps:
                    print("[Train] Reached max_train_steps, stopping early.", flush=True)
                    break
    finally:
        teacher_hook.close()
        student_hook.close()
        gpu_monitor.write_metrics()

    return global_step

def train_kd(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    student: nn.Module,
    teacher: nn.Module,
    trainable_params: Sequence[nn.Parameter],
    dataset: Dataset,
    gpu_monitor: Optional[GPUMonitor] = None,
) -> int:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    input_device = detect_input_device(student, device)
    print(f"[Device] Input tensors will be placed on: {input_device}", flush=True)

    collator = KDCollator(tokenizer=tokenizer, max_length=args.max_length)
    dataloader = build_dataloader(dataset, collator, args, shuffle=True)
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        foreach=False,
    )
    optimizer.zero_grad(set_to_none=True)
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_steps = args.max_train_steps or (args.num_epochs * num_update_steps_per_epoch)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    loss_path = os.path.join(args.output_dir, "loss_history.jsonl")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if os.path.exists(loss_path):
        os.remove(loss_path)

    print(
        f"[Train] KD strategy={args.strategy} epochs={args.num_epochs} "
        f"batches/epoch={len(dataloader)} total_steps={total_steps}",
        flush=True,
    )
    teacher.eval()
    student.train()
    global_step = 0
    start_time = time.time()

    monitor_context = (
        gpu_monitor.segment("training_loop") if gpu_monitor is not None else build_gpu_monitor(args)
    )
    with monitor_context:
        for epoch in range(args.num_epochs):
            epoch_loss = 0.0
            steps_in_epoch = 0
            for step, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(input_device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(input_device, non_blocking=True)

                with torch.no_grad():
                    teacher_out = teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                    teacher_logits = teacher_out.logits

                student_out = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                raw_loss = kd_loss_token_level(
                    teacher_logits=teacher_logits,
                    student_logits=student_out.logits,
                    attention_mask=attention_mask,
                    temperature=args.temperature,
                )
                loss = raw_loss / args.gradient_accumulation_steps
                loss.backward()

                epoch_loss += raw_loss.detach().float().item()
                steps_in_epoch += 1

                should_step = (
                    (step + 1) % args.gradient_accumulation_steps == 0
                    or (step + 1) == len(dataloader)
                )
                if not should_step:
                    continue

                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm, foreach=False)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                current_loss = raw_loss.detach().float().item()
                current_lr = lr_scheduler.get_last_lr()[0]
                append_loss(
                    loss_path,
                    {
                        "event": "step",
                        "time": time.time(),
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "total_steps": total_steps,
                        "loss": current_loss,
                        "lr": current_lr,
                    },
                )
                if global_step % args.logging_steps == 0:
                    elapsed = max(time.time() - start_time, 1e-8)
                    steps_per_sec = global_step / elapsed
                    remaining_steps = max(total_steps - global_step, 0)
                    eta_seconds = remaining_steps / max(steps_per_sec, 1e-8)
                    print(
                        f"[Step {global_step}/{total_steps}] loss={current_loss:.6f} "
                        f"lr={current_lr:.8f} {steps_per_sec:.3f} step/s "
                        f"ETA {int(eta_seconds // 3600):02d}:"
                        f"{int((eta_seconds % 3600) // 60):02d}:"
                        f"{int(eta_seconds % 60):02d}",
                        flush=True,
                    )

                if args.max_train_steps is not None and global_step >= args.max_train_steps:
                    break

            avg_epoch_loss = epoch_loss / max(1, steps_in_epoch)
            append_loss(
                loss_path,
                {
                    "event": "epoch",
                    "time": time.time(),
                    "epoch": epoch + 1,
                    "avg_loss": avg_epoch_loss,
                    "global_step": global_step,
                },
            )
            print(f"[Epoch {epoch + 1}/{args.num_epochs}] avg_loss={avg_epoch_loss:.6f}", flush=True)

            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                print("[Train] Reached max_train_steps, stopping early.", flush=True)
                break

    return global_step


def train_causal_lm(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    model: nn.Module,
    trainable_params: Sequence[nn.Parameter],
    dataset: Dataset,
    gpu_monitor: Optional[GPUMonitor] = None,
) -> int:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    input_device = detect_input_device(model, device)
    print(f"[Device] Input tensors will be placed on: {input_device}", flush=True)

    collator = CausalLMCollator(tokenizer=tokenizer, max_length=args.max_length)
    dataloader = build_dataloader(dataset, collator, args, shuffle=True)
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        foreach=False,
    )
    optimizer.zero_grad(set_to_none=True)
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_steps = args.max_train_steps or (args.num_epochs * num_update_steps_per_epoch)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    loss_path = os.path.join(args.output_dir, "loss_history.jsonl")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if os.path.exists(loss_path):
        os.remove(loss_path)

    print(
        f"[Train] CausalLM strategy={args.strategy} epochs={args.num_epochs} "
        f"batches/epoch={len(dataloader)} total_steps={total_steps}",
        flush=True,
    )
    model.train()
    global_step = 0
    start_time = time.time()

    monitor_context = (
        gpu_monitor.segment("training_loop") if gpu_monitor is not None else build_gpu_monitor(args)
    )
    with monitor_context:
        for epoch in range(args.num_epochs):
            epoch_loss = 0.0
            steps_in_epoch = 0
            for step, batch in enumerate(dataloader):
                batch = {key: value.to(input_device, non_blocking=True) for key, value in batch.items()}
                outputs = model(**batch, use_cache=False)
                raw_loss = outputs.loss
                if raw_loss is None:
                    raise RuntimeError("Model did not return a loss.")
                loss = raw_loss / args.gradient_accumulation_steps
                loss.backward()

                epoch_loss += raw_loss.detach().float().item()
                steps_in_epoch += 1
                should_step = (
                    (step + 1) % args.gradient_accumulation_steps == 0
                    or (step + 1) == len(dataloader)
                )
                if not should_step:
                    continue

                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm, foreach=False)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                current_loss = raw_loss.detach().float().item()
                current_lr = lr_scheduler.get_last_lr()[0]
                append_loss(
                    loss_path,
                    {
                        "event": "step",
                        "time": time.time(),
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "total_steps": total_steps,
                        "loss": current_loss,
                        "lr": current_lr,
                    },
                )
                if global_step % args.logging_steps == 0:
                    elapsed = max(time.time() - start_time, 1e-8)
                    steps_per_sec = global_step / elapsed
                    remaining_steps = max(total_steps - global_step, 0)
                    eta_seconds = remaining_steps / max(steps_per_sec, 1e-8)
                    print(
                        f"[Step {global_step}/{total_steps}] loss={current_loss:.6f} "
                        f"lr={current_lr:.8f} {steps_per_sec:.3f} step/s "
                        f"ETA {int(eta_seconds // 3600):02d}:"
                        f"{int((eta_seconds % 3600) // 60):02d}:"
                        f"{int(eta_seconds % 60):02d}",
                        flush=True,
                    )

                if args.max_train_steps is not None and global_step >= args.max_train_steps:
                    break

            avg_epoch_loss = epoch_loss / max(1, steps_in_epoch)
            append_loss(
                loss_path,
                {
                    "event": "epoch",
                    "time": time.time(),
                    "epoch": epoch + 1,
                    "avg_loss": avg_epoch_loss,
                    "global_step": global_step,
                },
            )
            print(f"[Epoch {epoch + 1}/{args.num_epochs}] avg_loss={avg_epoch_loss:.6f}", flush=True)

            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                print("[Train] Reached max_train_steps, stopping early.", flush=True)
                break

    return global_step


def run(args: argparse.Namespace) -> None:
    if args.strategy not in STRATEGIES:
        raise ValueError(f"Unsupported strategy: {args.strategy}")
    if args.output_dir and not Path(args.output_dir).resolve().is_relative_to(Path(args.workspace_root).resolve()):
        raise ValueError("output_dir must stay inside workspace_root.")
    if args.gpu_log_csv and not Path(args.gpu_log_csv).resolve().is_relative_to(Path(args.workspace_root).resolve()):
        raise ValueError("gpu_log_csv must stay inside workspace_root.")
    if args.gpu_metrics_json and not Path(args.gpu_metrics_json).resolve().is_relative_to(Path(args.workspace_root).resolve()):
        raise ValueError("gpu_metrics_json must stay inside workspace_root.")

    set_reproducibility(args.seed, deterministic=args.deterministic)
    prepare_output_dir(args.output_dir, allow_overwrite=args.allow_overwrite)

    dataset = TextLineDataset(args.calib_dataset_path, max_samples=args.max_calib_samples)
    source_path = source_checkpoint_path(args)
    if not source_path:
        raise ValueError("Either --student_model_name_or_path or --model_name_or_path is required.")

    teacher_strategies = {
        "direct_router_logit_matching",
        "only_router_kd",
        "router_top8_expert_kd",
        "router_top16_expert_kd",
        "router_top50_expert_kd",
        "router_top128_expert_kd",
        "full_kd",
    }
    router_only_kd_strategies = {"only_router_kd"}
    subset_kd_strategies = {"router_top8_expert_kd", "router_top16_expert_kd", "router_top50_expert_kd"}
    all_expert_kd_strategies = {"router_top128_expert_kd"}
    full_kd_strategies = {"full_kd"}
    router_only_ft_strategies = {"only_router_finetune"}
    subset_ft_strategies = {
        "router_top8_expert_finetune",
        "router_top16_expert_finetune",
        "router_top50_expert_finetune",
    }
    all_expert_ft_strategies = {"router_top128_expert_finetune"}
    full_ft_strategies = {"full_finetune"}

    if args.strategy in teacher_strategies:
        if not args.teacher_model_name_or_path:
            raise ValueError("--teacher_model_name_or_path is required for teacher-supervised strategies.")
        tokenizer = load_tokenizer(args.teacher_model_name_or_path)
        student = load_causal_lm(source_path, args, trainable=True)

        selected_summary: Optional[List[Dict[str, object]]] = None
        subset_gpu_monitor: Optional[GPUMonitor] = None
        if args.strategy == "direct_router_logit_matching":
            mark_all_frozen(student)
            enable_router_training(student)
            trainable_params, trainable_names = summarize_trainable_parameters(student)
        elif args.strategy in router_only_kd_strategies:
            mark_all_frozen(student)
            enable_router_training(student)
            trainable_params, trainable_names = summarize_trainable_parameters(student)
        elif args.strategy in subset_kd_strategies:
            selection_collator = KDCollator(tokenizer=tokenizer, max_length=args.max_length)
            selection_loader = build_dataloader(dataset, selection_collator, args, shuffle=False)
            input_device = detect_input_device(
                student,
                torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"),
            )
            subset_gpu_monitor = build_gpu_monitor(args)
            with subset_gpu_monitor.segment("expert_selection"):
                selected_by_prefix, selected_summary = select_top_physical_experts_per_layer(
                    model=student,
                    dataloader=selection_loader,
                    device=input_device,
                    experts_per_layer=args.experts_per_layer,
                    selection_top_k=args.selection_top_k,
                    logging_steps=args.selection_logging_steps,
                )
            mark_all_frozen(student)
            enable_router_training(student)
            enable_selected_expert_training(student, selected_by_prefix)
            trainable_params, trainable_names = summarize_trainable_parameters(
                student,
                allowed_selected_experts=selected_by_prefix,
            )
            save_json(os.path.join(args.output_dir, "selected_experts.json"), {"layers": selected_summary})
        elif args.strategy in all_expert_kd_strategies:
            mark_all_frozen(student)
            enable_router_training(student)
            selected_by_prefix, selected_summary, _ = enable_all_expert_training(student)
            trainable_params, trainable_names = summarize_trainable_parameters(
                student,
                allowed_selected_experts=selected_by_prefix,
            )
            save_json(os.path.join(args.output_dir, "selected_experts.json"), {"layers": selected_summary})
        elif args.strategy in full_kd_strategies:
            for param in student.parameters():
                param.requires_grad = True
            trainable_params, trainable_names = summarize_trainable_parameters(student)
            selected_summary = None
        else:
            raise ValueError(f"Unsupported teacher-supervised strategy: {args.strategy}")

        if hasattr(student, "config"):
            student.config.pad_token_id = tokenizer.pad_token_id
            student.config.use_cache = False

        teacher = load_causal_lm(args.teacher_model_name_or_path, args, trainable=False)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        try:
            if args.strategy == "direct_router_logit_matching":
                train_direct_router_logit_matching(
                    args,
                    tokenizer,
                    student,
                    teacher,
                    trainable_params,
                    dataset,
                )
            else:
                train_kd(args, tokenizer, student, teacher, trainable_params, dataset, subset_gpu_monitor)
        finally:
            if subset_gpu_monitor is not None:
                subset_gpu_monitor.write_metrics()
        del teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        save_model(student, tokenizer, args, trainable_names, selected_summary)
        del student

    else:
        fine_tune_strategies = (
            router_only_ft_strategies
            | subset_ft_strategies
            | all_expert_ft_strategies
            | full_ft_strategies
        )
        if args.strategy not in fine_tune_strategies:
            raise ValueError(f"Unsupported fine-tuning strategy: {args.strategy}")
        tokenizer = load_tokenizer(source_path)
        model = load_causal_lm(source_path, args, trainable=True)
        if hasattr(model, "config"):
            model.config.pad_token_id = tokenizer.pad_token_id
            model.config.use_cache = False

        subset_gpu_monitor: Optional[GPUMonitor] = None
        if args.strategy in router_only_ft_strategies:
            mark_all_frozen(model)
            enable_router_training(model)
            trainable_params, trainable_names = summarize_trainable_parameters(model)
            selected_summary = None
        elif args.strategy in subset_ft_strategies:
            selection_collator = KDCollator(tokenizer=tokenizer, max_length=args.max_length)
            selection_loader = build_dataloader(dataset, selection_collator, args, shuffle=False)
            input_device = detect_input_device(
                model,
                torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"),
            )
            subset_gpu_monitor = build_gpu_monitor(args)
            with subset_gpu_monitor.segment("expert_selection"):
                selected_by_prefix, selected_summary = select_top_physical_experts_per_layer(
                    model=model,
                    dataloader=selection_loader,
                    device=input_device,
                    experts_per_layer=args.experts_per_layer,
                    selection_top_k=args.selection_top_k,
                    logging_steps=args.selection_logging_steps,
                )
            mark_all_frozen(model)
            enable_router_training(model)
            enable_selected_expert_training(model, selected_by_prefix)
            trainable_params, trainable_names = summarize_trainable_parameters(
                model,
                allowed_selected_experts=selected_by_prefix,
            )
            save_json(os.path.join(args.output_dir, "selected_experts.json"), {"layers": selected_summary})
        elif args.strategy in all_expert_ft_strategies:
            mark_all_frozen(model)
            enable_router_training(model)
            selected_by_prefix, selected_summary, _ = enable_all_expert_training(model)
            trainable_params, trainable_names = summarize_trainable_parameters(
                model,
                allowed_selected_experts=selected_by_prefix,
            )
            save_json(os.path.join(args.output_dir, "selected_experts.json"), {"layers": selected_summary})
        elif args.strategy in full_ft_strategies:
            for param in model.parameters():
                param.requires_grad = True
            trainable_params, trainable_names = summarize_trainable_parameters(model)
            selected_summary = None
        else:
            raise ValueError(f"Unsupported fine-tuning strategy: {args.strategy}")

        try:
            train_causal_lm(args, tokenizer, model, trainable_params, dataset, subset_gpu_monitor)
        finally:
            if subset_gpu_monitor is not None:
                subset_gpu_monitor.write_metrics()
        save_model(model, tokenizer, args, trainable_names, selected_summary)
        del model

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    print(f"[Done] strategy={args.strategy} ratio={args.ratio}", flush=True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REAP recovery strategy trainer.")
    parser.add_argument("--strategy", required=True, choices=sorted(STRATEGIES))
    parser.add_argument("--ratio", required=True)
    parser.add_argument("--workspace_root", required=True)
    parser.add_argument("--teacher_model_name_or_path", default=None)
    parser.add_argument("--student_model_name_or_path", default=None)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--calib_dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_calib_samples", type=int, default=3000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_memory_per_gpu_gb", type=int, default=110)
    parser.add_argument("--max_shard_size", type=str, default="10GB")
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--allow_overwrite", action="store_true")
    parser.add_argument("--save_trainable_state", action="store_true")
    parser.add_argument("--experts_per_layer", type=int, default=8)
    parser.add_argument("--selection_top_k", type=int, default=8)
    parser.add_argument("--selection_logging_steps", type=int, default=20)
    parser.add_argument("--assigned_gpu_ids", default=None)
    parser.add_argument("--gpu_log_csv", required=True)
    parser.add_argument("--gpu_metrics_json", required=True)
    parser.add_argument("--gpu_log_interval", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
