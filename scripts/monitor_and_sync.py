#!/usr/bin/env python3
"""
monitor_and_sync.py — GRPO Training Watchdog, Milestone HF Sync & User Reporter

Monitors the Qwen3-VL-2B GRPO training job:
1. Status Reporting: sends a formatted report to Telegram every 60 minutes.
2. Milestone Sync: every 50 steps (or whenever milestone checkpoints occur),
   uploads adapter checkpoints and current logs to Hugging Face Hub.
3. Completion Handler: merges (if needed), publishes full run and merged model
   to Hugging Face Hub, and sends final summary to Telegram.
4. Fault Detection: alerts immediately if the training process dies.
"""

import os
import sys
import time
import json
import re
import glob
import shutil
import zipfile
import subprocess
import urllib.request
import urllib.parse
import requests
from typing import Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path

# Paths
BASE_DIR = Path("/workspace/qwen3-VL-2B-finetune")
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = OUTPUT_DIR / "logs/grpo"
TRAIN_LOG = LOG_DIR / "train.log"
GPU_CSV = LOG_DIR / "gpu.csv"
CONFIG_TXT = LOG_DIR / "config.txt"
CMD_TXT = LOG_DIR / "cmd.txt"
CHECKPOINTS_DIR = Path(os.environ.get("CHECKPOINTS_DIR", str(OUTPUT_DIR / "grpo_lora")))
MERGED_DIR = Path(os.environ.get("MERGED_DIR", str(OUTPUT_DIR / "grpo_merged")))
MODEL_BASE = os.environ.get("MODEL_BASE", "shahedm2001/qwen3-vl-2b-cataract-sft-stage2")
STATE_FILE = Path(os.environ.get("STATE_FILE", str(OUTPUT_DIR / ".sync_state.json")))
DATASET_LABEL = os.environ.get("DATASET_LABEL", "dataset_grpo")
TB_OFFSET = int(os.environ.get("TB_OFFSET", "0"))
TB_MERGE_DEST = Path(os.environ["TB_MERGE_DEST"]) if "TB_MERGE_DEST" in os.environ else None
WATCHDOG_LOG = OUTPUT_DIR / "logs/watchdog.log"
PYTHON_BIN = BASE_DIR / ".venv/bin/python"

# Hugging Face Configuration
MODEL_REPO = "shahedm2001/qwen3-vl-2b-cataract-grpo"
RUNS_REPO = "shahedm2001/qwen3-vl-2b-cataract-grpo-runs"

# Interval settings
REPORT_INTERVAL_SECONDS = 3600  # 60 minutes
LOOP_SLEEP_SECONDS = 30         # 30 seconds poll


def log_watchdog(msg: str):
    """Write message to stdout and watchdog.log."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{now_str}] {msg}"
    print(line, flush=True)
    try:
        WATCHDOG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(WATCHDOG_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"Failed writing to watchdog.log: {e}", file=sys.stderr)


def get_telegram_credentials():
    """Retrieve Telegram Bot Token and Chat ID."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("HERMES_SESSION_CHAT_ID", "").strip() or os.environ.get("TELEGRAM_HOME_CHANNEL", "").strip()

    if not token and os.path.exists("/root/.hermes/.env"):
        try:
            with open("/root/.hermes/.env", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("TELEGRAM_BOT_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip("\"'")
                    elif line.startswith("TELEGRAM_HOME_CHANNEL=") and not chat_id:
                        chat_id = line.split("=", 1)[1].strip().strip("\"'")
        except Exception as e:
            log_watchdog(f"Warning: could not read /root/.hermes/.env: {e}")

    if not chat_id:
        chat_id = "52775645"  # Fallback to Shahed's Telegram Chat ID

    return token, chat_id


def send_telegram(text: str, parse_mode: Optional[str] = "Markdown") -> bool:
    """Send message to user via Telegram Bot API."""
    token, chat_id = get_telegram_credentials()
    if not token or not chat_id:
        log_watchdog("Telegram credentials not configured; skipping Telegram send.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            if res_data.get("ok"):
                msg_id = res_data.get("result", {}).get("message_id")
                log_watchdog(f"Telegram message successfully delivered (id={msg_id}).")
                return True
            else:
                log_watchdog(f"Telegram API returned error: {res_data}")
    except Exception as e:
        log_watchdog(f"Failed sending Telegram message ({parse_mode}): {e}")
        # Retry once without Markdown formatting in case of formatting syntax errors
        if parse_mode:
            return send_telegram(text, parse_mode=None)

    return False


def send_telegram_document(file_path: Path, caption: Optional[str] = None) -> bool:
    """Send document/file to user via Telegram Bot API."""
    token, chat_id = get_telegram_credentials()
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    try:
        with open(file_path, "rb") as f:
            files = {"document": (file_path.name, f)}
            data = {"chat_id": chat_id}
            if caption:
                data["caption"] = caption
                data["parse_mode"] = "Markdown"
            resp = requests.post(url, data=data, files=files, timeout=60)
            res_json = resp.json()
            if res_json.get("ok"):
                msg_id = res_json.get("result", {}).get("message_id")
                log_watchdog(f"Telegram document {file_path.name} sent successfully (id={msg_id}).")
                return True
            else:
                log_watchdog(f"Telegram sendDocument error: {res_json}")
    except Exception as e:
        log_watchdog(f"Failed sending Telegram document: {e}")
    return False


def create_logs_zip_with_timestamp() -> Path:
    """Create a zip of all logs with dataset and timestamp in the filename."""
    dataset_name = DATASET_LABEL
    time_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    zip_name = f"logs_{dataset_name}_{time_str}.zip"
    zip_path = OUTPUT_DIR / zip_name
    zip_logs(zip_path)
    return zip_path


def create_tensorboard_zip_with_timestamp() -> Optional[Path]:
    """Create a zip of TensorBoard data (raw .tfevents + exported CSVs) with dataset and timestamp."""
    dataset_name = DATASET_LABEL
    time_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    zip_name = f"tensorboard_{dataset_name}_{time_str}.zip"
    zip_path = OUTPUT_DIR / zip_name

    runs_dir = CHECKPOINTS_DIR / "runs"
    if not runs_dir.exists():
        return None

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            # 1. Add raw event files
            for ef in runs_dir.rglob("*events*"):
                if ef.is_file():
                    zf.write(ef, arcname=f"raw_events/{ef.name}")

            # 2. Query TensorBoard REST API on localhost:16006 if available
            try:
                url_tags = "http://localhost:16006/data/plugin/scalars/tags"
                req = urllib.request.Request(url_tags, headers={"User-Agent": "Watchdog"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    tags_data = json.loads(resp.read().decode())

                if tags_data:
                    run_name = list(tags_data.keys())[0]
                    tags = list(tags_data[run_name].keys())

                    all_step_data = {}
                    all_steps = set()

                    for tag in sorted(tags):
                        encoded_tag = urllib.parse.quote(tag, safe="")
                        encoded_run = urllib.parse.quote(run_name, safe="")
                        url_scalar = f"http://localhost:16006/data/plugin/scalars/scalars?tag={encoded_tag}&run={encoded_run}"
                        try:
                            req_s = urllib.request.Request(url_scalar, headers={"User-Agent": "Watchdog"})
                            with urllib.request.urlopen(req_s, timeout=5) as resp_s:
                                pts = json.loads(resp_s.read().decode())
                            short_name = tag.replace("train/", "")

                            csv_lines = ["wall_time,step," + short_name]
                            for pt in pts:
                                w_t, stp, val = pt[0], pt[1], pt[2]
                                csv_lines.append(f"{w_t},{stp},{val}")
                                if stp not in all_step_data:
                                    all_step_data[stp] = {}
                                all_step_data[stp][short_name] = val
                                all_steps.add(stp)

                            zf.writestr(f"exported_csvs/csv_metrics/{short_name}.csv", "\n".join(csv_lines))
                        except Exception:
                            pass

                    if all_steps:
                        sorted_steps = sorted(all_steps)
                        short_names = sorted(list({k for d in all_step_data.values() for k in d.keys()}))
                        header = ["step"] + short_names
                        master_rows = [",".join(header)]
                        for stp in sorted_steps:
                            row = [str(stp)] + [str(all_step_data[stp].get(name, "")) for name in short_names]
                            master_rows.append(",".join(row))
                        zf.writestr("exported_csvs/all_scalars_by_step.csv", "\n".join(master_rows))
            except Exception as e:
                log_watchdog(f"Note: Could not query TensorBoard API: {e}")

        return zip_path
    except Exception as e:
        log_watchdog(f"Error creating tensorboard zip: {e}")
        return None


def get_hf_api():
    """Import and return HfApi instance with token."""
    try:
        import huggingface_hub
        token = huggingface_hub.get_token() or os.environ.get("HF_TOKEN")
        return huggingface_hub.HfApi(token=token), token
    except Exception as e:
        log_watchdog(f"Error importing huggingface_hub: {e}")
        return None, None


def load_sync_state() -> dict:
    """Load tracking state from .sync_state.json."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log_watchdog(f"Warning: could not load {STATE_FILE}: {e}")

    return {
        "uploaded_milestones": [],
        "uploaded_checkpoints": [],
        "last_report_time": 0,
        "last_reported_step": 0,
        "completed": False,
        "start_time": time.time(),
    }


def save_sync_state(state: dict):
    """Save tracking state to .sync_state.json."""
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        tmp.replace(STATE_FILE)
    except Exception as e:
        log_watchdog(f"Error saving {STATE_FILE}: {e}")


def is_train_grpo_alive() -> bool:
    """Check if train_grpo.py process is alive."""
    try:
        res = subprocess.run(["pgrep", "-f", "train_grpo.py"], capture_output=True, text=True)
        return res.returncode == 0 and bool(res.stdout.strip())
    except Exception:
        return False


def parse_train_log():
    """Extract progress bar stats and recent step metrics from train.log."""
    if not TRAIN_LOG.exists():
        return None

    # Read last ~200KB of train.log
    size = TRAIN_LOG.stat().st_size
    read_bytes = min(size, 204800)
    try:
        with open(TRAIN_LOG, "rb") as f:
            if size > read_bytes:
                f.seek(size - read_bytes)
            content = f.read().decode("utf-8", errors="replace")
    except Exception as e:
        log_watchdog(f"Error reading {TRAIN_LOG}: {e}")
        return None

    lines = content.splitlines()

    # If Stage 4 banner exists, only look at content after the last Stage 4 start
    if "[=== STARTING TRAINING STAGE 4" in content:
        stage_part = content.rsplit("[=== STARTING TRAINING STAGE 4", 1)[1]
        lines = stage_part.splitlines()

    # 1. Progress bar match: e.g. "  2/532 [10:23<45:42:34, 310.48s/it]"
    progress_info = {
        "step": 0,
        "total_steps": 532,
        "pct": 0.0,
        "elapsed": "unknown",
        "eta": "unknown",
        "rate": "unknown",
    }
    prog_re = re.compile(r"(\d+)/(\d+)\s+\[([0-9:]+)<([0-9:]+),\s*([0-9.]+(?:s/it|it/s))\]")
    for line in reversed(lines):
        if "Loading weights" in line or "weights:" in line:
            continue
        m = prog_re.search(line)
        if m:
            step = int(m.group(1))
            total = int(m.group(2))
            if total > 532:
                continue
            progress_info["step"] = step
            progress_info["total_steps"] = total
            progress_info["pct"] = (step / total) * 100.0 if total > 0 else 0.0
            progress_info["elapsed"] = m.group(3)
            progress_info["eta"] = m.group(4)
            progress_info["rate"] = m.group(5)
            break

    # 2. Extract RL micro-slices and Trainer loss steps
    all_slices = []
    latest_trainer = {}
    import ast

    for line in lines:
        if "'reward_mean':" in line or '"reward_mean":' in line:
            start = line.find("{")
            end = line.rfind("}")
            if start != -1 and end > start:
                try:
                    all_slices.append(ast.literal_eval(line[start:end+1]))
                except Exception:
                    pass
        elif "'loss':" in line and "'grad_norm':" in line:
            start = line.find("{")
            end = line.rfind("}")
            if start != -1 and end > start:
                try:
                    latest_trainer = ast.literal_eval(line[start:end+1])
                except Exception:
                    pass

    latest_rl = all_slices[-1] if all_slices else {}
    step_slices = all_slices[-8:] if len(all_slices) >= 8 else all_slices

    def safe_float(val, default=0.0):
        try:
            return float(val)
        except Exception:
            return default

    step_agg = {
        "num_slices": len(step_slices),
        "mean_rewards": [safe_float(s.get("reward_mean")) for s in step_slices],
        "step_mean_reward": (sum(safe_float(s.get("reward_mean")) for s in step_slices) / len(step_slices)) if step_slices else 0.0,
        "step_exact_match": (sum(safe_float(s.get("fraction_reward_one")) for s in step_slices) / len(step_slices)) if step_slices else 0.0,
        "step_zero_reward": (sum(safe_float(s.get("fraction_reward_zero")) for s in step_slices) / len(step_slices)) if step_slices else 0.0,
        "zero_std_count": sum(1 for s in step_slices if safe_float(s.get("zero_std_group_fraction")) > 0.5),
        "step_approx_kl": (sum(safe_float(s.get("approx_kl")) for s in step_slices) / len(step_slices)) if step_slices else 0.0,
        "step_comp_len": (sum(safe_float(s.get("comp_len_mean")) for s in step_slices) / len(step_slices)) if step_slices else 0.0,
        "step_entropy": (sum(safe_float(s.get("entropy_proxy")) for s in step_slices) / len(step_slices)) if step_slices else 0.0,
    }

    return {
        "progress": progress_info,
        "rl": latest_rl,
        "step_agg": step_agg,
        "trainer": latest_trainer,
    }


def parse_gpu_stats():
    """Extract latest GPU metrics and peak VRAM from gpu.csv."""
    if not GPU_CSV.exists():
        return {
            "used_mib": 0,
            "total_mib": 24576,
            "temp": 0,
            "power": 0.0,
            "peak_mib": 0,
            "util_pct": 0,
        }

    latest = {}
    peak_mib = 0
    try:
        with open(GPU_CSV, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 7:
                    try:
                        # timestamp,gpu_index,gpu_util_percent,memory_used_mib,memory_total_mib,temp,power
                        util = float(parts[2])
                        used = int(parts[3])
                        total = int(parts[4])
                        temp = int(parts[5])
                        power = float(parts[6])

                        if used > peak_mib:
                            peak_mib = used

                        latest = {
                            "used_mib": used,
                            "total_mib": total,
                            "temp": temp,
                            "power": power,
                            "peak_mib": peak_mib,
                            "util_pct": util,
                        }
                    except ValueError:
                        continue
    except Exception as e:
        log_watchdog(f"Error parsing {GPU_CSV}: {e}")

    latest["peak_mib"] = peak_mib
    return latest


def format_status_message(parsed_log, gpu_stats) -> str:
    """Build clean Markdown status report."""
    prog = parsed_log["progress"] if parsed_log else {}
    rl = parsed_log["rl"] if parsed_log else {}
    step_agg = parsed_log.get("step_agg", {}) if parsed_log else {}
    trainer = parsed_log["trainer"] if parsed_log else {}

    step = prog.get("step", 0)
    total = prog.get("total_steps", 532)
    pct = prog.get("pct", 0.0)
    eta = prog.get("eta", "unknown")
    rate = prog.get("rate", "unknown")
    elapsed = prog.get("elapsed", "unknown")

    epoch = trainer.get("epoch", rl.get("epoch", "0"))
    if isinstance(epoch, (float, int)):
        epoch_str = f"{epoch:.4f}"
    else:
        epoch_str = str(epoch)

    trainer_loss = trainer.get("loss", "N/A")
    grad_norm = trainer.get("grad_norm", "N/A")
    lr = trainer.get("learning_rate", "N/A")

    # Aggregate step metrics across all 8 micro-slices
    step_reward_mean = step_agg.get("step_mean_reward", 0.0)
    step_exact_match = step_agg.get("step_exact_match", 0.0)
    zero_std_count = step_agg.get("zero_std_count", 0)
    num_slices = step_agg.get("num_slices", 8)
    prompt_rewards = step_agg.get("mean_rewards", [])
    prompt_rewards_str = ", ".join(f"{r:.2f}" for r in prompt_rewards) if prompt_rewards else "N/A"
    step_kl = step_agg.get("step_approx_kl", 0.0)
    step_comp_len = step_agg.get("step_comp_len", 0.0)

    used_gb = gpu_stats.get("used_mib", 0) / 1024.0
    total_gb = gpu_stats.get("total_mib", 24576) / 1024.0
    peak_gb = gpu_stats.get("peak_mib", 0) / 1024.0
    temp = gpu_stats.get("temp", 0)
    power = gpu_stats.get("power", 0.0)

    # Format the latest individual micro-slice dictionary
    rl_clean = {k: v for k, v in rl.items()}
    rl_dict_str = json.dumps(rl_clean, indent=None)

    msg = (
        f"📊 *Qwen3-VL GRPO Training Update*\n\n"
        f"• *Progress:* Step `{step}/{total}` ({pct:.1f}%) | Epoch `{epoch_str}`\n"
        f"• *Speed & ETA:* `{rate}` | Elapsed: `{elapsed}` | ETA: `{eta}`\n\n"
        f"1️⃣ *Optimizer Step (The Short Log)*\n"
        f"• *Accumulated Loss:* `{trainer_loss}`\n"
        f"• *Gradient Norm:* `{grad_norm}`\n"
        f"• *Learning Rate:* `{lr}`\n\n"
        f"2️⃣ *Full Step Rollout Aggregate (8 Prompts / 32 Completions)*\n"
        f"• *Step Mean Reward:* `{step_reward_mean:.4f}`\n"
        f"• *Step Exact Match Rate (1.0):* `{step_exact_match * 100.0:.1f}%`\n"
        f"• *Per-Prompt Rewards:* `[{prompt_rewards_str}]`\n"
        f"• *Zero-Variance Groups:* `{zero_std_count}/{num_slices}` prompts\n"
        f"• *Step Approx KL:* `{step_kl:.6f}` | Avg Comp Len: `{step_comp_len:.1f}` tokens\n\n"
        f"3️⃣ *Latest Micro-Slice Rollout (The Long Log Sample)*\n"
        f"```json\n{rl_dict_str}\n```\n\n"
        f"4️⃣ *Hardware & VRAM*\n"
        f"• *VRAM Allocation:* `{used_gb:.1f} / {total_gb:.1f} GB` (Peak: `{peak_gb:.1f} GB`)\n"
        f"• *GPU Temp:* `{temp}°C` | *Power Draw:* `{power:.1f} W`\n"
    )
    return msg


def zip_logs(dest_zip: Path) -> bool:
    """Archive current log files."""
    try:
        dest_zip.parent.mkdir(parents=True, exist_ok=True)
        files_to_zip = [TRAIN_LOG, GPU_CSV, CONFIG_TXT, CMD_TXT]
        with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for f in files_to_zip:
                if f.exists():
                    z.write(f, arcname=f.name)
            # Also include any summary or loss csv
            for extra in LOG_DIR.glob("*.csv"):
                if extra != GPU_CSV and extra.exists():
                    z.write(extra, arcname=extra.name)
        return True
    except Exception as e:
        log_watchdog(f"Error creating logs zip: {e}")
        return False


def get_latest_checkpoint() -> Tuple[Optional[Path], int]:
    """Find the latest checkpoint directory and step number."""
    if not CHECKPOINTS_DIR.exists():
        return None, 0

    ckpts = []
    for d in CHECKPOINTS_DIR.glob("checkpoint-*"):
        if d.is_dir():
            m = re.match(r"checkpoint-(\d+)", d.name)
            if m:
                ckpts.append((int(m.group(1)), d))

    if not ckpts:
        return None, 0

    ckpts.sort(key=lambda x: x[0])
    latest_step, latest_path = ckpts[-1]
    return latest_path, latest_step


def is_checkpoint_ready(checkpoint_path: Path) -> bool:
    """Check if checkpoint directory is complete and finished writing."""
    req_files = ["adapter_model.safetensors", "adapter_config.json", "trainer_state.json"]
    for rf in req_files:
        f = checkpoint_path / rf
        if not f.exists() or f.stat().st_size == 0:
            return False
    return True


def upload_milestone(checkpoint_path: Path, step: int, milestone: int) -> Optional[str]:
    """Upload checkpoint and logs to Hugging Face under a distinct dataset+timestamped folder (no overwriting)."""
    api, token = get_hf_api()
    if not api or not token:
        log_watchdog("Cannot upload milestone: Hugging Face API not available.")
        return None

    log_watchdog(f"Starting upload for milestone {milestone} (using {checkpoint_path.name})...")

    # Verify essential checkpoint files
    req_files = ["adapter_model.safetensors", "adapter_config.json"]
    for rf in req_files:
        if not (checkpoint_path / rf).exists():
            log_watchdog(f"Checkpoint {checkpoint_path.name} is missing {rf}; skipping upload.")
            return None

    try:
        dataset_name = DATASET_LABEL
        time_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        folder_name = f"{checkpoint_path.name}_{dataset_name}_{time_str}"
        remote_ckpt_path = f"checkpoints/{folder_name}"

        # Sanitize auto-generated checkpoint README.md to ensure valid Hub base_model
        ckpt_readme = checkpoint_path / "README.md"
        if ckpt_readme.exists():
            try:
                rm_text = ckpt_readme.read_text(encoding="utf-8")
                if "/workspace" in rm_text:
                    rm_text = re.sub(r"base_model:\s*/workspace[^\n]+", f"base_model: {MODEL_REPO}", rm_text)
                    rm_text = re.sub(r"base_model:adapter:/workspace[^\n]+", f"base_model:adapter:{MODEL_REPO}", rm_text)
                    ckpt_readme.write_text(rm_text, encoding="utf-8")
                    log_watchdog(f"Sanitized base_model metadata in {checkpoint_path.name}/README.md")
            except Exception as e:
                log_watchdog(f"Warning sanitizing README: {e}")

        log_watchdog(f"Uploading checkpoint files to {MODEL_REPO}/{remote_ckpt_path}...")
        api.upload_folder(
            folder_path=str(checkpoint_path),
            path_in_repo=remote_ckpt_path,
            repo_id=MODEL_REPO,
            repo_type="model",
            commit_message=f"Add GRPO adapter {folder_name} (step {step})",
        )
        log_watchdog(f"Checkpoint folder {folder_name} uploaded successfully.")

        # Archive and upload logs into the same folder and training_logs
        zip_path = OUTPUT_DIR / f"training_logs_{step}.zip"
        if zip_logs(zip_path):
            api.upload_file(
                path_or_fileobj=str(zip_path),
                path_in_repo=f"{remote_ckpt_path}/logs.zip",
                repo_id=MODEL_REPO,
                repo_type="model",
                commit_message=f"Sync training logs into {folder_name}",
            )
            api.upload_file(
                path_or_fileobj=str(zip_path),
                path_in_repo=f"training_logs/logs_{dataset_name}_{time_str}.zip",
                repo_id=MODEL_REPO,
                repo_type="model",
                commit_message=f"Sync training logs archive at step {step}",
            )
            if CONFIG_TXT.exists():
                api.upload_file(
                    path_or_fileobj=str(CONFIG_TXT),
                    path_in_repo=f"{remote_ckpt_path}/config.txt",
                    repo_id=MODEL_REPO,
                    repo_type="model",
                    commit_message=f"Add config.txt to {folder_name}",
                )
            if CMD_TXT.exists():
                api.upload_file(
                    path_or_fileobj=str(CMD_TXT),
                    path_in_repo=f"{remote_ckpt_path}/cmd.txt",
                    repo_id=MODEL_REPO,
                    repo_type="model",
                    commit_message=f"Add cmd.txt to {folder_name}",
                )
            try:
                zip_path.unlink()
            except Exception:
                pass

        # Merge TensorBoard events if configured
        if TB_MERGE_DEST:
            try:
                from scripts.merge_tensorboard import merge_tb_runs
                merge_tb_runs(str(CHECKPOINTS_DIR / "runs"), str(TB_MERGE_DEST), step_offset=TB_OFFSET)
            except Exception as e:
                log_watchdog(f"Error merging TensorBoard on milestone: {e}")

        # Archive and upload TensorBoard data into the same folder and tensorboard_data
        tb_zip = create_tensorboard_zip_with_timestamp()
        if tb_zip and tb_zip.exists():
            log_watchdog(f"Uploading TensorBoard data archive {tb_zip.name} to Hugging Face...")
            api.upload_file(
                path_or_fileobj=str(tb_zip),
                path_in_repo=f"{remote_ckpt_path}/tensorboard.zip",
                repo_id=MODEL_REPO,
                repo_type="model",
                commit_message=f"Sync TensorBoard data into {folder_name}",
            )
            api.upload_file(
                path_or_fileobj=str(tb_zip),
                path_in_repo=f"tensorboard_data/{tb_zip.name}",
                repo_id=MODEL_REPO,
                repo_type="model",
                commit_message=f"Sync TensorBoard data archive at step {step}",
            )
            try:
                tb_zip.unlink()
            except Exception:
                pass

        log_watchdog(f"Milestone {milestone} successfully uploaded to folder {folder_name} on Hugging Face!")
        return folder_name
    except Exception as e:
        log_watchdog(f"Failed uploading milestone to Hugging Face: {e}")
        return None


def run_completion_tasks(state: dict):
    """Handle end-of-training tasks: merge verification, full upload, and final report."""
    log_watchdog("Training finished. Starting completion tasks...")

    # 1. Check if output/grpo_merged exists. If not, wait briefly for grpo_train.sh to finish merge
    max_wait = 600  # 10 minutes
    waited = 0
    while not (MERGED_DIR / "config.json").exists() and waited < max_wait:
        log_watchdog(f"Waiting for merged model at {MERGED_DIR}... ({waited}s)")
        time.sleep(15)
        waited += 15

    # If merge still not done, run merge manually
    if not (MERGED_DIR / "config.json").exists():
        log_watchdog("Running merge_lora.py manually...")
        cmd_merge = [
            str(PYTHON_BIN),
            "src/merge_lora.py",
            "--model-path", str(CHECKPOINTS_DIR),
            "--model-base", MODEL_BASE,
            "--save-model-path", str(MERGED_DIR),
            "--safe-serialization",
        ]
        res = subprocess.run(cmd_merge, cwd=str(BASE_DIR), capture_output=True, text=True)
        log_watchdog(f"merge_lora stdout: {res.stdout}")
        if res.returncode != 0:
            log_watchdog(f"merge_lora stderr: {res.stderr}")

    # Merge TensorBoard events if configured
    if TB_MERGE_DEST:
        try:
            log_watchdog(f"Running TensorBoard event merge with offset {TB_OFFSET} into {TB_MERGE_DEST}...")
            from scripts.merge_tensorboard import merge_tb_runs
            merge_tb_runs(str(CHECKPOINTS_DIR / "runs"), str(TB_MERGE_DEST), step_offset=TB_OFFSET)
        except Exception as e:
            log_watchdog(f"Error merging TensorBoard events: {e}")

    # 2. Run upload_run_to_hub.py
    api, token = get_hf_api()
    if token:
        os.environ["HF_TOKEN"] = token

    upload_cmd = [
        str(PYTHON_BIN),
        "src/upload_run_to_hub.py",
        "--output-dir", "output",
        "--merged-dir", str(MERGED_DIR.relative_to(BASE_DIR)),
        "--runs-repo", RUNS_REPO,
        "--model-repo", MODEL_REPO,
    ]
    log_watchdog(f"Running {' '.join(upload_cmd)}...")
    res = subprocess.run(upload_cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
    log_watchdog(f"upload_run_to_hub output:\n{res.stdout}")
    if res.returncode != 0:
        log_watchdog(f"upload_run_to_hub error:\n{res.stderr}")

    # 3. Collect final statistics
    parsed_log = parse_train_log()
    gpu_stats = parse_gpu_stats()
    rl = parsed_log["rl"] if parsed_log else {}
    reward_mean = rl.get("reward_mean", "N/A")
    peak_gb = gpu_stats.get("peak_mib", 0) / 1024.0

    start_time = state.get("start_time", time.time())
    total_wall_sec = time.time() - start_time
    hours = int(total_wall_sec // 3600)
    minutes = int((total_wall_sec % 3600) // 60)

    final_msg = (
        f"🏆 *Qwen3-VL GRPO Training Finished!*\n\n"
        f"• *Total Wall Time:* `{hours}h {minutes}m`\n"
        f"• *Peak VRAM:* `{peak_gb:.1f} GB`\n"
        f"• *Final Reward Mean:* `{reward_mean}`\n\n"
        f"🔗 *Hugging Face Links:*\n"
        f"• [Model Checkpoint & Merged Model](https://huggingface.co/{MODEL_REPO})\n"
        f"• [Full Training Run & Logs Archive](https://huggingface.co/datasets/{RUNS_REPO})\n"
    )

    send_telegram(final_msg)
    state["completed"] = True
    save_sync_state(state)
    log_watchdog("All completion tasks finished successfully.")


def main():
    log_watchdog("Starting monitor_and_sync daemon...")
    state = load_sync_state()

    # Initial check & confirmation
    parsed_log = parse_train_log()
    gpu_stats = parse_gpu_stats()
    alive = is_train_grpo_alive()

    init_msg = (
        f"🚀 *Qwen3-VL GRPO Monitor & Sync Active*\n\n"
        f"Watchdog daemon has been initialized.\n"
        f"• Process status: `{'Running (PID found)' if alive else 'Not Running'}`\n"
    )
    if parsed_log and parsed_log.get("progress"):
        p = parsed_log["progress"]
        init_msg += f"• Current Step: `{p.get('step', 0)}/{p.get('total_steps', 532)}` ({p.get('pct', 0.0):.1f}%)\n"
        init_msg += f"• Pace: `{p.get('rate', 'N/A')}` | ETA: `{p.get('eta', 'N/A')}`\n"

    init_msg += (
        f"\n📡 *Reporting Policy:*\n"
        f"- Regular status updates: every 60 minutes\n"
        f"- Checkpoint sync to Hugging Face: every 20 steps\n"
        f"- Automatic final merge & upload on completion"
    )
    send_telegram(init_msg)
    state["last_report_time"] = time.time()
    save_sync_state(state)

    failure_alerted = False

    while True:
        try:
            time.sleep(LOOP_SLEEP_SECONDS)
            now = time.time()

            alive = is_train_grpo_alive()
            parsed_log = parse_train_log()
            gpu_stats = parse_gpu_stats()

            curr_step = parsed_log["progress"]["step"] if (parsed_log and parsed_log.get("progress")) else 0

            # ── Check for Unexpected Death ───────────────────────────
            if not alive and not state.get("completed", False):
                # Check if it finished all 532 steps
                total_steps = parsed_log["progress"]["total_steps"] if parsed_log else 532
                if curr_step >= total_steps or curr_step >= 530:
                    log_watchdog("Training reached target steps and exited.")
                    run_completion_tasks(state)
                    break
                else:
                    # Check exit code or tail of train.log
                    log_watchdog("ALERT: train_grpo.py is not running before target steps!")
                    if not failure_alerted:
                        tail_lines = ""
                        if TRAIN_LOG.exists():
                            try:
                                with open(TRAIN_LOG, "r", encoding="utf-8", errors="replace") as f:
                                    all_lines = f.readlines()
                                    tail_lines = "".join(all_lines[-30:])
                            except Exception:
                                pass

                        alert_msg = (
                            f"⚠️ *ALERT: GRPO Training Process Terminated Unexpectedly!*\n\n"
                            f"Last recorded step: `{curr_step}`\n\n"
                            f"*Last 30 lines of train.log:*\n```\n{tail_lines[-3000:]}\n```"
                        )
                        send_telegram(alert_msg)
                        failure_alerted = True

                    # Sleep longer before rechecking
                    time.sleep(60)
                    continue

            # ── Task A: Periodic Status Report (Every 60 min) ────────
            if now - state.get("last_report_time", 0) >= REPORT_INTERVAL_SECONDS:
                status_text = format_status_message(parsed_log, gpu_stats)
                log_watchdog("Sending periodic status report to Telegram...")
                send_telegram(status_text)

                # Generate zipped logs with dataset and timestamp and send to Telegram
                try:
                    logs_zip = create_logs_zip_with_timestamp()
                    log_watchdog(f"Sending logs archive {logs_zip.name} to Telegram...")
                    send_telegram_document(logs_zip, caption=f"📁 Training Logs: `{logs_zip.name}`")
                except Exception as e:
                    log_watchdog(f"Failed sending logs zip to Telegram: {e}")

                # Generate zipped TensorBoard data with dataset and timestamp and send to Telegram
                try:
                    tb_zip = create_tensorboard_zip_with_timestamp()
                    if tb_zip and tb_zip.exists():
                        log_watchdog(f"Sending TensorBoard archive {tb_zip.name} to Telegram...")
                        send_telegram_document(tb_zip, caption=f"📊 TensorBoard Data: `{tb_zip.name}`")
                except Exception as e:
                    log_watchdog(f"Failed sending TensorBoard zip to Telegram: {e}")

                state["last_report_time"] = now
                state["last_reported_step"] = curr_step
                save_sync_state(state)

            # ── Task B: Milestone Upload (Every 20 Steps) ────────────
            # Target milestones: 20, 40, 60, 80, 100, 120, ...
            total_steps = parsed_log.get("progress", {}).get("total_steps", 532) if parsed_log else 532
            milestones = [m for m in range(20, total_steps, 20)]
            for ms in milestones:
                if curr_step >= ms and ms not in state.get("uploaded_milestones", []):
                    target_ckpt_path = CHECKPOINTS_DIR / f"checkpoint-{ms}"
                    if target_ckpt_path.exists() and is_checkpoint_ready(target_ckpt_path):
                        log_watchdog(f"Milestone {ms} reached & verified on disk! Starting synchronized upload for step {ms}...")
                        uploaded_folder = upload_milestone(target_ckpt_path, curr_step, ms)
                        if uploaded_folder:
                            state.setdefault("uploaded_milestones", []).append(ms)
                            state.setdefault("uploaded_checkpoints", []).append(uploaded_folder)
                            save_sync_state(state)

                            milestone_msg = (
                                f"📦 *Hugging Face Milestone Checkpoint Uploaded*\n\n"
                                f"• *Milestone:* Step `{ms}` (current training step: `{curr_step}`)\n"
                                f"• *Destination Folder:* `{uploaded_folder}`\n"
                                f"• *Uploaded Files:* LoRA adapter weights, vision merger (`non_lora_state_dict.bin`), `logs.zip`, `tensorboard.zip`, and configs\n"
                                f"• *Repository:* [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO})\n"
                                f"• *Note:* Synced strictly at milestone step `{ms}`. Previous checkpoints remain preserved."
                            )
                            send_telegram(milestone_msg)
                    else:
                        log_watchdog(f"Step {curr_step} reached milestone {ms}, waiting for checkpoint-{ms} to finish saving to disk...")

        except Exception as e:
            log_watchdog(f"Watchdog loop exception: {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()
