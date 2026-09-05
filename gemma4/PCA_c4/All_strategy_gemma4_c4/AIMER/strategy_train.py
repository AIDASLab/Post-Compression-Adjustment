#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train/calibrate Gemma4 AIMER-pruned checkpoints with comparable recovery strategies."""

import argparse
import csv
import gc
import inspect
import json
import math
import os
import random
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover - depends on the runtime transformers version.
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForCausalLM
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None


STRATEGIES = {
    "direct_router_logit_matching",
    "only_router_kd",
    "router_top8_expert_kd",
    "router_top16_expert_kd",
    "router_top50_expert_kd",
    "router_top128_expert_kd",
    "full_kd",
    "only_router_finetune",
    "router_top8_expert_finetune",
    "router_top16_expert_finetune",
    "router_top50_expert_finetune",
    "router_top128_expert_finetune",
    "full_finetune",
}

KD_STRATEGIES = {
    "only_router_kd",
    "router_top8_expert_kd",
    "router_top16_expert_kd",
    "router_top50_expert_kd",
    "router_top128_expert_kd",
    "full_kd",
}

FINETUNE_STRATEGIES = {
    "only_router_finetune",
    "router_top8_expert_finetune",
    "router_top16_expert_finetune",
    "router_top50_expert_finetune",
    "router_top128_expert_finetune",
    "full_finetune",
}

ROUTER_ONLY_STRATEGIES = {
    "only_router_kd",
    "only_router_finetune",
}

ALL_EXPERT_STRATEGIES = {
    "router_top128_expert_kd",
    "router_top128_expert_finetune",
}

FULL_MODEL_STRATEGIES = {
    "full_kd",
    "full_finetune",
}

EXPERT_SUBSET_SIZE = {
    "router_top8_expert_kd": 8,
    "router_top8_expert_finetune": 8,
    "router_top16_expert_kd": 16,
    "router_top16_expert_finetune": 16,
    "router_top50_expert_kd": 50,
    "router_top50_expert_finetune": 50,
}

SIDECAR_FILES = (
    "retained_expert_indices.json",
    "pruned_experts.json",
    "expert_mapping.json",
    "pruning_plan.json",
    "calib_free_metadata.json",
    "gemma4_aimer_metadata.json",
    "calib_free_scores.csv",
    "calib_free_score_table.csv",
    "sanity_check_report.json",
    "processor_config.json",
    "generation_config.json",
    "chat_template.jinja",
)


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


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
                    if max_samples is not None and len(self.samples) >= max_samples:
                        break
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
                    if max_samples is not None and len(self.samples) >= max_samples:
                        break
        else:
            raise ValueError("Unsupported dataset file type. Use .txt or .json")

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

    @contextmanager
    def segment(self, segment_name: str):
        self.start(segment_name)
        try:
            yield self
        finally:
            self.stop()
            self.write_metrics()

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
            f"[GPU] Monitoring started GPUs={','.join(self.gpu_ids) or 'auto'} "
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
        print(f"[GPU] Monitoring stopped segment={self._current_segment}.", flush=True)

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


def selected_summary_to_prefix_map(
    selected_experts: Optional[List[Dict[str, object]]],
) -> Optional[Dict[str, List[int]]]:
    if not selected_experts:
        return None
    selected_by_prefix: Dict[str, List[int]] = {}
    for entry in selected_experts:
        prefix = entry.get("moe_prefix")
        selected_ids = entry.get("selected_physical_experts", [])
        if not isinstance(prefix, str) or not isinstance(selected_ids, list):
            continue
        selected_by_prefix[prefix] = [int(expert_id) for expert_id in selected_ids]
    return selected_by_prefix or None


def count_effective_masked_trainable_parameters(
    model: nn.Module,
    selected_experts_by_prefix: Optional[Dict[str, List[int]]],
) -> int:
    if not selected_experts_by_prefix:
        return count_parameters(model, only_trainable=True)

    module_dict = dict(model.named_modules())
    selected_expert_tensors = set()
    effective_count = 0
    for prefix, selected_ids in selected_experts_by_prefix.items():
        experts = module_dict.get(prefix + ".experts")
        if experts is None:
            raise RuntimeError(f"Selected expert module not found: {prefix}.experts")
        physical_num_experts = int(getattr(experts, "physical_num_experts", getattr(experts, "num_experts", 0)))
        selected_count = len({int(expert_id) for expert_id in selected_ids})
        for param_name in ("gate_up_proj", "down_proj"):
            param = getattr(experts, param_name, None)
            if param is None:
                raise RuntimeError(f"Packed expert tensor missing: {prefix}.experts.{param_name}")
            tensor_name = f"{prefix}.experts.{param_name}"
            selected_expert_tensors.add(tensor_name)
            if not param.requires_grad:
                continue
            if physical_num_experts > 0 and param.dim() > 0 and int(param.shape[0]) == physical_num_experts:
                effective_count += selected_count * (param.numel() // physical_num_experts)
            else:
                effective_count += param.numel()

    router_prefixes = tuple(prefix + ".router." for prefix in find_gemma4_moe_prefixes(model))
    for name, param in model.named_parameters():
        if not param.requires_grad or name in selected_expert_tensors:
            continue
        if name.startswith(router_prefixes):
            effective_count += param.numel()
    return effective_count


def detect_input_device(model: nn.Module, fallback: torch.device) -> torch.device:
    try:
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            for param in embeddings.parameters():
                return param.device
    except Exception:
        pass
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


def get_model_loader():
    if AutoModelForImageTextToText is not None:
        return AutoModelForImageTextToText
    if AutoModelForCausalLM is not None:
        return AutoModelForCausalLM
    raise ImportError(
        "This script requires Transformers with AutoModelForImageTextToText or AutoModelForCausalLM."
    )


def load_gemma4_model(path: str, args: argparse.Namespace, trainable: bool) -> nn.Module:
    print(f"[Init] Loading model: {path}", flush=True)
    model_cls = get_model_loader()
    try:
        model = model_cls.from_pretrained(path, **build_model_load_kwargs(args, trainable=trainable))
    except Exception as exc:
        raise RuntimeError(
            "Failed to load Gemma4/AIMER model. Make sure a compatible Gemma4 "
            "environment has a Transformers build with Gemma4 support."
        ) from exc
    if args.cpu or torch.cuda.device_count() <= 1:
        device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
        model.to(device)
    if trainable and args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    set_model_cache(model, use_cache=False)
    return model


def set_model_cache(model: nn.Module, use_cache: bool) -> None:
    if hasattr(model, "config"):
        try:
            model.config.use_cache = use_cache
        except Exception:
            pass
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None:
            try:
                text_config.use_cache = use_cache
            except Exception:
                pass


def set_pad_token_id(model: nn.Module, pad_token_id: Optional[int]) -> None:
    if pad_token_id is None or not hasattr(model, "config"):
        return
    try:
        model.config.pad_token_id = pad_token_id
    except Exception:
        pass
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None:
        try:
            text_config.pad_token_id = pad_token_id
        except Exception:
            pass


def find_gemma4_moe_prefixes(model: nn.Module) -> List[str]:
    prefixes: List[str] = []
    for module_name, module in model.named_modules():
        if not getattr(module, "enable_moe_block", False):
            continue
        if not hasattr(module, "router") or not hasattr(module, "experts"):
            continue
        prefixes.append(module_name)
    if not prefixes:
        raise RuntimeError("No Gemma4 sparse MoE layers were found.")
    print(f"[MoE] Found {len(prefixes)} sparse Gemma4 MoE layer(s).", flush=True)
    return prefixes


def mark_all_frozen(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def enable_router_training(model: nn.Module) -> List[str]:
    prefixes = find_gemma4_moe_prefixes(model)
    router_prefixes = tuple(prefix + ".router." for prefix in prefixes)
    router_names: List[str] = []
    for name, param in model.named_parameters():
        if name.startswith(router_prefixes):
            param.requires_grad = True
            router_names.append(name)
    if not router_names:
        raise RuntimeError("No Gemma4 router parameters were found.")
    print(f"[Router] Enabled {len(router_names)} router tensor(s).", flush=True)
    return router_names


def parse_layer_index(prefix: str) -> int:
    parts = prefix.split(".")
    for idx, part in enumerate(parts):
        if part == "layers" and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                return -1
    return -1


def get_logical_to_physical(module: nn.Module, device: torch.device) -> torch.Tensor:
    mapping = getattr(module.experts, "expert_id_mapping", None)
    if mapping is None:
        logical = int(getattr(module.experts, "logical_num_experts", getattr(module.experts, "num_experts", 0)))
        return torch.arange(logical, dtype=torch.long, device=device)
    return mapping.to(device=device, dtype=torch.long)


class RouterProjHookCollector:
    def __init__(self, module_dict: Dict[str, nn.Module], prefixes: Sequence[str], detach: bool = True):
        self.prefixes = list(prefixes)
        self.detach = detach
        self.values: Dict[str, torch.Tensor] = {}
        self.handles = []
        for prefix in self.prefixes:
            proj_name = prefix + ".router.proj"
            proj_module = module_dict.get(proj_name)
            if proj_module is None:
                raise RuntimeError(f"Gemma4 router projection not found: {proj_name}")
            self.handles.append(proj_module.register_forward_hook(self._make_hook(prefix)))

    def _make_hook(self, prefix: str):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            self.values[prefix] = output.detach() if self.detach else output

        return hook

    def clear(self) -> None:
        self.values.clear()

    def collect(self) -> Dict[str, torch.Tensor]:
        missing = [prefix for prefix in self.prefixes if prefix not in self.values]
        if missing:
            raise RuntimeError(f"Router hook did not observe every sparse layer: {missing[:5]}")
        return dict(self.values)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def router_scores_for_topk(module: nn.Module, logits: torch.Tensor) -> torch.Tensor:
    # Gemma4TextRouter selects top-k from softmax(router.proj(hidden_states)).
    # per_expert_scale is applied only after top-k selection, so it must not affect
    # the expert selection counts used by router+top-k strategies.
    return torch.softmax(logits.float(), dim=-1)


def flatten_logits_with_mask(logits: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.dim() == 3:
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_mask = attention_mask.reshape(-1).bool().to(flat_logits.device)
        return flat_logits, flat_mask
    if logits.dim() == 2:
        flat_logits = logits
        flat_mask = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
        if attention_mask.numel() == logits.shape[0]:
            flat_mask = attention_mask.reshape(-1).bool().to(logits.device)
        return flat_logits, flat_mask
    raise RuntimeError(f"Unexpected router logits shape: {tuple(logits.shape)}")


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

    prefixes = find_gemma4_moe_prefixes(model)
    module_dict = dict(model.named_modules())
    first_module = module_dict[prefixes[0]]
    physical_num_experts = int(getattr(first_module.experts, "physical_num_experts", first_module.experts.num_experts))
    logical_num_experts = int(getattr(first_module.experts, "logical_num_experts", physical_num_experts))
    router_top_k = selection_top_k or int(getattr(model.config.text_config, "top_k_experts", 1))
    router_top_k = max(1, min(router_top_k, logical_num_experts))

    counts = [torch.zeros(physical_num_experts, dtype=torch.long) for _ in prefixes]
    total_valid_tokens = 0
    start_time = time.time()
    hook_collector = RouterProjHookCollector(module_dict, prefixes)

    print(
        f"[Select] Selecting top physical experts experts_per_layer={experts_per_layer} "
        f"router_top_k={router_top_k}",
        flush=True,
    )

    model.eval()
    try:
        for step, batch in enumerate(dataloader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            total_valid_tokens += int(attention_mask.reshape(-1).bool().sum().item())

            hook_collector.clear()
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

            router_outputs = hook_collector.collect()
            for layer_idx, prefix in enumerate(prefixes):
                layer = module_dict[prefix]
                router_scores = router_scores_for_topk(layer, router_outputs[prefix])
                flat_scores, flat_mask = flatten_logits_with_mask(router_scores, attention_mask)
                valid_scores = flat_scores[flat_mask]
                if valid_scores.numel() == 0:
                    continue
                logical_ids = torch.topk(valid_scores, k=router_top_k, dim=-1).indices
                mapping = get_logical_to_physical(layer, logical_ids.device)
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


def make_expert_mask_hook(selected_ids: Sequence[int]):
    selected = torch.tensor(sorted(set(int(idx) for idx in selected_ids)), dtype=torch.long)

    def hook(grad: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(grad.shape[0], dtype=grad.dtype, device=grad.device)
        mask[selected.to(grad.device)] = 1
        view_shape = [grad.shape[0]] + [1] * (grad.dim() - 1)
        return grad * mask.view(*view_shape)

    return hook


def enable_selected_expert_training(
    model: nn.Module,
    selected_experts_by_prefix: Dict[str, List[int]],
) -> List[str]:
    if not hasattr(model, "_aimer_gradient_hooks"):
        model._aimer_gradient_hooks = []

    module_dict = dict(model.named_modules())
    trainable_names: List[str] = []
    for prefix, selected_ids in selected_experts_by_prefix.items():
        experts = module_dict.get(prefix + ".experts")
        if experts is None:
            raise RuntimeError(f"Selected expert module not found: {prefix}.experts")
        physical_num_experts = int(getattr(experts, "physical_num_experts", experts.num_experts))
        bad_ids = [idx for idx in selected_ids if idx < 0 or idx >= physical_num_experts]
        if bad_ids:
            raise RuntimeError(f"Selected expert ids out of range for {prefix}: {bad_ids}")
        for param_name in ("gate_up_proj", "down_proj"):
            param = getattr(experts, param_name, None)
            if param is None:
                raise RuntimeError(f"Packed expert tensor missing: {prefix}.experts.{param_name}")
            param.requires_grad = True
            model._aimer_gradient_hooks.append(param.register_hook(make_expert_mask_hook(selected_ids)))
            trainable_names.append(f"{prefix}.experts.{param_name}")

    print(
        f"[Experts] Enabled masked training for {len(selected_experts_by_prefix)} layer(s), "
        f"{len(trainable_names)} packed expert tensor(s).",
        flush=True,
    )
    return trainable_names


def enable_all_expert_training(model: nn.Module) -> List[str]:
    prefixes = find_gemma4_moe_prefixes(model)
    module_dict = dict(model.named_modules())
    trainable_names: List[str] = []
    for prefix in prefixes:
        experts = module_dict.get(prefix + ".experts")
        if experts is None:
            raise RuntimeError(f"Expert module not found: {prefix}.experts")
        for param_name in ("gate_up_proj", "down_proj"):
            param = getattr(experts, param_name, None)
            if param is None:
                raise RuntimeError(f"Packed expert tensor missing: {prefix}.experts.{param_name}")
            param.requires_grad = True
            trainable_names.append(f"{prefix}.experts.{param_name}")
    print(
        f"[Experts] Enabled all expert tensor training for {len(prefixes)} layer(s), "
        f"{len(trainable_names)} packed expert tensor(s).",
        flush=True,
    )
    return trainable_names


def summarize_trainable_parameters(
    model: nn.Module,
    allowed_selected_experts: Optional[Dict[str, List[int]]] = None,
) -> Tuple[List[nn.Parameter], List[str]]:
    trainable_params: List[nn.Parameter] = []
    trainable_names: List[str] = []
    unexpected: List[str] = []
    selected_expert_tensors = set()
    if allowed_selected_experts:
        for prefix in allowed_selected_experts:
            selected_expert_tensors.add(prefix + ".experts.gate_up_proj")
            selected_expert_tensors.add(prefix + ".experts.down_proj")

    router_prefixes = tuple(prefix + ".router." for prefix in find_gemma4_moe_prefixes(model))
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        trainable_params.append(param)
        trainable_names.append(name)
        is_router = name.startswith(router_prefixes)
        is_selected_expert_tensor = name in selected_expert_tensors
        if allowed_selected_experts is not None and not (is_router or is_selected_expert_tensor):
            unexpected.append(name)

    if unexpected:
        raise RuntimeError(
            "Found trainable parameters outside router + selected expert tensor scope:\n"
            + "\n".join(unexpected[:50])
        )
    if not trainable_params:
        raise RuntimeError("No trainable parameters found.")

    packed_trainable_count = sum(param.numel() for param in trainable_params)
    total_count = count_parameters(model, only_trainable=False)
    if allowed_selected_experts is not None:
        effective_trainable_count = count_effective_masked_trainable_parameters(
            model,
            allowed_selected_experts,
        )
        print(
            f"[Params] Trainable params: {effective_trainable_count:,} / {total_count:,} "
            f"({100.0 * effective_trainable_count / max(1, total_count):.4f}%) "
            "[effective masked expert slices]",
            flush=True,
        )
        print(
            f"[Params] Packed optimizer tensor params: {packed_trainable_count:,} / {total_count:,} "
            f"({100.0 * packed_trainable_count / max(1, total_count):.4f}%)",
            flush=True,
        )
    else:
        print(
            f"[Params] Trainable params: {packed_trainable_count:,} / {total_count:,} "
            f"({100.0 * packed_trainable_count / max(1, total_count):.4f}%)",
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




def resolved_experts_per_layer(args: argparse.Namespace) -> int:
    value = args.experts_per_layer
    if value is None:
        value = EXPERT_SUBSET_SIZE.get(args.strategy)
    if value is None or value <= 0:
        raise ValueError(f"Could not resolve a positive expert subset size for strategy={args.strategy}.")
    return int(value)


def load_aimer_retained_expert_indices(source_dir: str) -> Dict[int, List[int]]:
    path = Path(source_dir) / "retained_expert_indices.json"
    if not path.is_file():
        path = Path(source_dir) / "expert_mapping.json"
        if not path.is_file():
            raise FileNotFoundError(
                "AIMER Direct Router Logit Matching requires retained_expert_indices.json "
                "or expert_mapping.json from the pruned student checkpoint."
            )
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        retained = {}
        for key, value in raw.items():
            if isinstance(value, dict) and "pruned_model_expert_id_to_original_expert_id" in value:
                inverse = value["pruned_model_expert_id_to_original_expert_id"]
                retained[int(key)] = [int(inverse[str(idx)]) for idx in range(len(inverse))]
            elif isinstance(value, dict) and "kept_original_expert_ids" in value:
                retained[int(key)] = [int(item) for item in value["kept_original_expert_ids"]]
            else:
                raise ValueError(f"Unsupported AIMER expert mapping entry for layer {key}: {type(value)}")
        return retained

    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(key): [int(item) for item in value] for key, value in raw.items()}


def collect_router_prefix_pairs(teacher: nn.Module, student: nn.Module) -> List[Tuple[str, str]]:
    teacher_prefixes = find_gemma4_moe_prefixes(teacher)
    student_prefixes = find_gemma4_moe_prefixes(student)
    if len(teacher_prefixes) != len(student_prefixes):
        raise RuntimeError(
            "Teacher/student sparse MoE layer counts do not match: "
            f"{len(teacher_prefixes)} vs {len(student_prefixes)}"
        )
    return list(zip(teacher_prefixes, student_prefixes))


def project_teacher_router_logits_to_aimer_student_space(
    student_prefix: str,
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    retained_original_ids_by_layer: Dict[int, List[int]],
) -> torch.Tensor:
    teacher_dim = int(teacher_logits.shape[-1])
    student_dim = int(student_logits.shape[-1])
    if teacher_dim == student_dim:
        return teacher_logits

    layer_idx = parse_layer_index(student_prefix)
    retained_original_ids = retained_original_ids_by_layer.get(layer_idx)
    if retained_original_ids is None:
        raise ValueError(f"No AIMER retained expert mapping found for layer {layer_idx}.")
    if len(retained_original_ids) != student_dim:
        raise ValueError(
            f"AIMER retained mapping length mismatch at layer {layer_idx}: "
            f"mapping={len(retained_original_ids)}, student_dim={student_dim}"
        )
    if max(retained_original_ids) >= teacher_dim or min(retained_original_ids) < 0:
        raise ValueError(
            f"AIMER retained mapping for layer {layer_idx} contains ids outside teacher router dim {teacher_dim}."
        )
    index = torch.tensor(retained_original_ids, dtype=torch.long, device=teacher_logits.device)
    return teacher_logits.index_select(dim=-1, index=index)


def router_logit_matching_loss(
    teacher_router_outputs: Dict[str, torch.Tensor],
    student_router_outputs: Dict[str, torch.Tensor],
    router_prefix_pairs: Sequence[Tuple[str, str]],
    retained_original_ids_by_layer: Dict[int, List[int]],
    attention_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    total_loss = None
    loss_device: Optional[torch.device] = None
    matched_layers = 0

    for teacher_prefix, student_prefix in router_prefix_pairs:
        teacher_logits = teacher_router_outputs[teacher_prefix]
        student_logits = student_router_outputs[student_prefix]
        teacher_target = project_teacher_router_logits_to_aimer_student_space(
            student_prefix=student_prefix,
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            retained_original_ids_by_layer=retained_original_ids_by_layer,
        )

        if temperature != 1.0:
            teacher_target = teacher_target / temperature
            student_logits = student_logits / temperature

        flat_teacher, teacher_mask = flatten_logits_with_mask(teacher_target, attention_mask)
        flat_student, student_mask = flatten_logits_with_mask(student_logits, attention_mask)
        if flat_teacher.shape != flat_student.shape:
            raise ValueError(
                f"Router logit shape mismatch at {student_prefix}: "
                f"teacher={tuple(flat_teacher.shape)}, student={tuple(flat_student.shape)}"
            )

        teacher_mask = teacher_mask.to(flat_student.device)
        student_mask = student_mask.to(flat_student.device)
        mask = (teacher_mask & student_mask).to(dtype=torch.float32)
        tokenwise_mse = (flat_student.float() - flat_teacher.to(flat_student.device).float()).pow(2).mean(dim=-1)
        layer_loss = (tokenwise_mse * mask).sum() / (mask.sum() + 1e-8)
        if loss_device is None:
            loss_device = layer_loss.device
        layer_loss = layer_loss.to(loss_device)
        total_loss = layer_loss if total_loss is None else total_loss + layer_loss
        matched_layers += 1

    if total_loss is None or matched_layers == 0:
        raise RuntimeError("No router layers were matched for Direct Router Logit Matching.")
    return total_loss / matched_layers


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


def save_aimer_sidecars(source_dir: str, output_dir: str) -> None:
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
        print(f"[Save] Copied sidecar file: {filename}", flush=True)


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
    save_aimer_sidecars(source_checkpoint_path(args), args.output_dir)

    selected_experts_by_prefix = selected_summary_to_prefix_map(selected_experts)
    packed_trainable_params = count_parameters(model, only_trainable=True)
    effective_trainable_params = count_effective_masked_trainable_parameters(
        model,
        selected_experts_by_prefix,
    )
    total_params = count_parameters(model, only_trainable=False)

    summary = {
        "ratio": args.ratio,
        "strategy": args.strategy,
        "all_experts_training": args.strategy in ALL_EXPERT_STRATEGIES,
        "compression_method": "AIMER",
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
        "experts_per_layer": args.experts_per_layer,
        "selection_top_k": args.selection_top_k,
        "router_logit_matching_mapping_policy": "teacher_original_logits_index_select_by_aimer_retained_expert_indices",
        "seed": args.seed,
        "deterministic": args.deterministic,
        "gradient_checkpointing": args.gradient_checkpointing,
        "max_memory_per_gpu_gb": args.max_memory_per_gpu_gb,
        "trainable_params": effective_trainable_params,
        "effective_trainable_params": effective_trainable_params,
        "packed_optimizer_trainable_params": packed_trainable_params,
        "total_params": total_params,
        "trainable_param_names": list(trainable_param_names),
        "selected_experts": selected_experts or [],
        "packed_expert_gradient_masking": bool(selected_experts),
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


def build_optimizer(trainable_params: Sequence[nn.Parameter], args: argparse.Namespace):
    return torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        foreach=False,
    )


def get_forward_kwargs(model: nn.Module) -> Dict[str, object]:
    kwargs: Dict[str, object] = {"use_cache": False}
    try:
        forward_sig = inspect.signature(model.forward)
        if "logits_to_keep" in forward_sig.parameters:
            kwargs["logits_to_keep"] = 0
    except Exception:
        pass
    return kwargs


def train_direct_router_logit_matching(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    student: nn.Module,
    teacher: nn.Module,
    trainable_params: Sequence[nn.Parameter],
    dataset: Dataset,
    retained_original_ids_by_layer: Dict[int, List[int]],
    gpu_monitor: Optional[GPUMonitor] = None,
) -> int:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    input_device = detect_input_device(student, device)
    print(f"[Device] Input tensors will be placed on: {input_device}", flush=True)

    collator = KDCollator(tokenizer=tokenizer, max_length=args.max_length)
    dataloader = build_dataloader(dataset, collator, args, shuffle=True)
    optimizer = build_optimizer(trainable_params, args)
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

    teacher_module_dict = dict(teacher.named_modules())
    student_module_dict = dict(student.named_modules())
    router_prefix_pairs = collect_router_prefix_pairs(teacher, student)
    teacher_prefixes = [teacher_prefix for teacher_prefix, _ in router_prefix_pairs]
    student_prefixes = [student_prefix for _, student_prefix in router_prefix_pairs]
    teacher_hooks = RouterProjHookCollector(teacher_module_dict, teacher_prefixes, detach=True)
    student_hooks = RouterProjHookCollector(student_module_dict, student_prefixes, detach=False)

    print(
        f"[Train] Direct Router Logit Matching epochs={args.num_epochs} "
        f"batches/epoch={len(dataloader)} total_steps={total_steps} "
        f"matched_router_layers={len(router_prefix_pairs)}",
        flush=True,
    )
    teacher.eval()
    student.train()
    global_step = 0
    start_time = time.time()
    forward_kwargs = get_forward_kwargs(student)
    teacher_forward_kwargs = get_forward_kwargs(teacher)

    monitor_context = (
        gpu_monitor.segment("training_loop") if gpu_monitor is not None else build_gpu_monitor(args).segment("training_loop")
    )
    try:
        with monitor_context:
            for epoch in range(args.num_epochs):
                epoch_loss = 0.0
                steps_in_epoch = 0
                for step, batch in enumerate(dataloader):
                    input_ids = batch["input_ids"].to(input_device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(input_device, non_blocking=True)

                    teacher_hooks.clear()
                    with torch.no_grad():
                        teacher(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            **teacher_forward_kwargs,
                        )
                        teacher_router_outputs = teacher_hooks.collect()

                    student_hooks.clear()
                    student(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **forward_kwargs,
                    )
                    student_router_outputs = student_hooks.collect()

                    raw_loss = router_logit_matching_loss(
                        teacher_router_outputs=teacher_router_outputs,
                        student_router_outputs=student_router_outputs,
                        router_prefix_pairs=router_prefix_pairs,
                        retained_original_ids_by_layer=retained_original_ids_by_layer,
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
                            "loss_type": "aimer_router_logit_mse",
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
                        "loss_type": "aimer_router_logit_mse",
                    },
                )
                print(f"[Epoch {epoch + 1}/{args.num_epochs}] avg_loss={avg_epoch_loss:.6f}", flush=True)

                if args.max_train_steps is not None and global_step >= args.max_train_steps:
                    print("[Train] Reached max_train_steps, stopping early.", flush=True)
                    break
    finally:
        teacher_hooks.close()
        student_hooks.close()

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
    optimizer = build_optimizer(trainable_params, args)
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
    forward_kwargs = get_forward_kwargs(student)
    teacher_forward_kwargs = get_forward_kwargs(teacher)

    monitor_context = (
        gpu_monitor.segment("training_loop") if gpu_monitor is not None else build_gpu_monitor(args).segment("training_loop")
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
                        **teacher_forward_kwargs,
                    )
                    teacher_logits = teacher_out.logits

                student_out = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **forward_kwargs,
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
    optimizer = build_optimizer(trainable_params, args)
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
    forward_kwargs = get_forward_kwargs(model)

    monitor_context = (
        gpu_monitor.segment("training_loop") if gpu_monitor is not None else build_gpu_monitor(args).segment("training_loop")
    )
    with monitor_context:
        for epoch in range(args.num_epochs):
            epoch_loss = 0.0
            steps_in_epoch = 0
            for step, batch in enumerate(dataloader):
                batch = {key: value.to(input_device, non_blocking=True) for key, value in batch.items()}
                outputs = model(**batch, **forward_kwargs)
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
    workspace_root = Path(args.workspace_root).resolve()
    for checked in (args.output_dir, args.gpu_log_csv, args.gpu_metrics_json):
        if checked and not is_relative_to(Path(checked), workspace_root):
            raise ValueError(f"Path must stay inside workspace_root: {checked}")

    if args.strategy in EXPERT_SUBSET_SIZE and args.experts_per_layer is None:
        args.experts_per_layer = EXPERT_SUBSET_SIZE[args.strategy]

    set_reproducibility(args.seed, deterministic=args.deterministic)
    prepare_output_dir(args.output_dir, allow_overwrite=args.allow_overwrite)

    dataset = TextLineDataset(args.calib_dataset_path, max_samples=args.max_calib_samples)
    source_path = source_checkpoint_path(args)
    if not source_path:
        raise ValueError("Either --student_model_name_or_path or --model_name_or_path is required.")

    if args.strategy == "direct_router_logit_matching":
        if not args.teacher_model_name_or_path:
            raise ValueError("--teacher_model_name_or_path is required for Direct Router Logit Matching.")
        tokenizer = load_tokenizer(args.teacher_model_name_or_path)
        student = load_gemma4_model(source_path, args, trainable=True)
        mark_all_frozen(student)
        enable_router_training(student)
        trainable_params, trainable_names = summarize_trainable_parameters(student)
        set_pad_token_id(student, tokenizer.pad_token_id)
        set_model_cache(student, use_cache=False)
        retained_original_ids_by_layer = load_aimer_retained_expert_indices(source_path)

        teacher = load_gemma4_model(args.teacher_model_name_or_path, args, trainable=False)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        set_pad_token_id(teacher, tokenizer.pad_token_id)
        set_model_cache(teacher, use_cache=False)
        train_direct_router_logit_matching(
            args,
            tokenizer,
            student,
            teacher,
            trainable_params,
            dataset,
            retained_original_ids_by_layer,
        )
        del teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        save_model(student, tokenizer, args, trainable_names, selected_experts=None)
        del student

    elif args.strategy in KD_STRATEGIES:
        if not args.teacher_model_name_or_path:
            raise ValueError("--teacher_model_name_or_path is required for KD strategies.")
        tokenizer = load_tokenizer(args.teacher_model_name_or_path)
        student = load_gemma4_model(source_path, args, trainable=True)

        selected_summary: Optional[List[Dict[str, object]]] = None
        gpu_monitor: Optional[GPUMonitor] = None
        if args.strategy in ROUTER_ONLY_STRATEGIES:
            mark_all_frozen(student)
            enable_router_training(student)
            trainable_params, trainable_names = summarize_trainable_parameters(student)
        elif args.strategy in FULL_MODEL_STRATEGIES:
            for param in student.parameters():
                param.requires_grad = True
            trainable_params, trainable_names = summarize_trainable_parameters(student)
        elif args.strategy in ALL_EXPERT_STRATEGIES:
            mark_all_frozen(student)
            enable_router_training(student)
            enable_all_expert_training(student)
            trainable_params, trainable_names = summarize_trainable_parameters(student)
        else:
            selection_collator = KDCollator(tokenizer=tokenizer, max_length=args.max_length)
            selection_loader = build_dataloader(dataset, selection_collator, args, shuffle=False)
            input_device = detect_input_device(
                student,
                torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"),
            )
            gpu_monitor = build_gpu_monitor(args)
            with gpu_monitor.segment("expert_selection"):
                selected_by_prefix, selected_summary = select_top_physical_experts_per_layer(
                    model=student,
                    dataloader=selection_loader,
                    device=input_device,
                    experts_per_layer=resolved_experts_per_layer(args),
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
        set_pad_token_id(student, tokenizer.pad_token_id)
        set_model_cache(student, use_cache=False)

        teacher = load_gemma4_model(args.teacher_model_name_or_path, args, trainable=False)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        set_pad_token_id(teacher, tokenizer.pad_token_id)
        set_model_cache(teacher, use_cache=False)
        try:
            train_kd(args, tokenizer, student, teacher, trainable_params, dataset, gpu_monitor)
        finally:
            if gpu_monitor is not None:
                gpu_monitor.write_metrics()
        del teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        save_model(student, tokenizer, args, trainable_names, selected_summary)
        del student

    elif args.strategy in FINETUNE_STRATEGIES:
        tokenizer = load_tokenizer(source_path)
        model = load_gemma4_model(source_path, args, trainable=True)
        set_pad_token_id(model, tokenizer.pad_token_id)
        set_model_cache(model, use_cache=False)

        selected_summary: Optional[List[Dict[str, object]]] = None
        gpu_monitor: Optional[GPUMonitor] = None
        if args.strategy in ROUTER_ONLY_STRATEGIES:
            mark_all_frozen(model)
            enable_router_training(model)
            trainable_params, trainable_names = summarize_trainable_parameters(model)
        elif args.strategy in FULL_MODEL_STRATEGIES:
            for param in model.parameters():
                param.requires_grad = True
            trainable_params, trainable_names = summarize_trainable_parameters(model)
        elif args.strategy in ALL_EXPERT_STRATEGIES:
            mark_all_frozen(model)
            enable_router_training(model)
            enable_all_expert_training(model)
            trainable_params, trainable_names = summarize_trainable_parameters(model)
        else:
            selection_collator = KDCollator(tokenizer=tokenizer, max_length=args.max_length)
            selection_loader = build_dataloader(dataset, selection_collator, args, shuffle=False)
            input_device = detect_input_device(
                model,
                torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"),
            )
            gpu_monitor = build_gpu_monitor(args)
            with gpu_monitor.segment("expert_selection"):
                selected_by_prefix, selected_summary = select_top_physical_experts_per_layer(
                    model=model,
                    dataloader=selection_loader,
                    device=input_device,
                    experts_per_layer=resolved_experts_per_layer(args),
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

        try:
            train_causal_lm(args, tokenizer, model, trainable_params, dataset, gpu_monitor)
        finally:
            if gpu_monitor is not None:
                gpu_monitor.write_metrics()
        save_model(model, tokenizer, args, trainable_names, selected_summary)
        del model

    else:
        raise ValueError(f"Unsupported strategy: {args.strategy}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    print(f"[Done] strategy={args.strategy} ratio={args.ratio}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma4 AIMER recovery strategy trainer.")
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
    parser.add_argument("--experts_per_layer", type=int, default=None)
    parser.add_argument("--selection_top_k", type=int, default=None)
    parser.add_argument("--selection_logging_steps", type=int, default=20)
    parser.add_argument("--assigned_gpu_ids", default=None)
    parser.add_argument("--gpu_log_csv", required=True)
    parser.add_argument("--gpu_metrics_json", required=True)
    parser.add_argument("--gpu_log_interval", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
