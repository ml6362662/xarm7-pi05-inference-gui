"""Strict RLinf pi05 checkpoint adapter; contains no robot SDK calls."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
from model_contract import (JOINT, TCP, EEFDELTA, contract_from_config,
                            normalize_tcp_actions, state_dim, action_dim)

TOKENIZER = ROOT / "assets" / "paligemma_tokenizer.model"


def verify_runtime():
    import importlib.metadata
    manifest = json.loads((ROOT / "runtime_manifest.json").read_text())
    for name, version in manifest["versions"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"推理依赖 {name} 的版本已变化，请重新进行离线验证")
    for filename, digest in manifest["source_hashes"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"已验证的推理代码发生变化：{filename}")
    if hashlib.sha256(TOKENIZER.read_bytes()).hexdigest() != manifest["tokenizer_sha256"]:
        raise RuntimeError("PaliGemma tokenizer 文件校验失败")


def read_bundle(filename):
    """Require the task-specific sidecars, not a generic base-model config."""
    import yaml

    path = Path(filename).expanduser().resolve(strict=True)
    if path.suffix != ".pt":
        raise ValueError("请选择完整的 .pt 模型文件")
    config = yaml.safe_load((path.parent / "training_config.yaml").read_text())
    model = config["actor"]["model"]
    contract = contract_from_config(config, path)
    ns, na = state_dim(contract["mode"]), action_dim(contract["mode"])
    if (
        model["action_dim"] != na
        or model["num_action_chunks"] != 50
        or model["openpi"].get("discrete_state_input") is not True
        or model.get("openpi_data", {}).get("pad_state") is not False
    ):
        raise ValueError(f"模型不是本次 XArm7 π0.5 的 {ns} 维状态、{na} 维动作、50 帧配置")
    if contract["mode"] == EEFDELTA and (
        model["openpi"].get("action_env_dim") != 7
        or model["openpi"].get("action_chunk") != 50
        or model["openpi"].get("max_token_len") != 200
    ):
        raise ValueError("EEFDelta 网络输出或状态token配置不匹配")
    if model["openpi"].get("num_images_in_input") != 2:
        raise ValueError("模型相机数量与双相机部署不匹配")
    stats = json.loads((path.parent / "norm_stats.json").read_text())["norm_stats"]
    for name, dim, active in (("state", ns, ns), ("actions", 32, na)):
        stats[name] = {k: np.asarray(v, dtype=np.float64) for k, v in stats[name].items()}
        for key in ("mean", "std", "q01", "q99"):
            a = stats[name][key]
            allowed_dims = (active, dim) if name == "actions" else (dim,)
            if a.shape not in tuple((d,) for d in allowed_dims) or not np.isfinite(a).all():
                raise ValueError(f"归一化参数无效：{name}.{key}")
        if np.any(stats[name]["q99"][:active] <= stats[name]["q01"][:active]):
            raise ValueError("有效动作/状态维度的归一化区间无效")
    verification = path.parent / "download_verified.json"
    if not verification.is_file():
        raise ValueError("缺少 download_verified.json，不能确认该文件下载完整")
    meta = json.loads(verification.read_text())
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    if meta.get("bytes") != path.stat().st_size or meta.get("sha256") != digest.hexdigest():
        raise ValueError("模型大小或 SHA-256 不匹配，拒绝加载未完成或被修改的权重")
    view = "full"
    if contract["mode"] == EEFDELTA:
        spec_path = path.parent / "dataset_spec.json"
        if not spec_path.exists():
            spec_path = path.parent.parent / "training_snapshot/dataset_spec.json"
        view = json.loads(spec_path.read_text()).get("policy_state_view", "full")
        expected = {"xarm348_state_b_20261112":"rot6d_gripper", "xarm348_state_c_20261112":"gripper"}.get(config.get("runner", {}).get("logger", {}).get("experiment_name"))
        if (expected is not None and view != expected) or view not in ("full","rot6d_gripper","gripper"):
            raise ValueError("Model and policy state view mismatch")
    meta = dict(meta, model_contract=contract, policy_state_view=view)
    return path, stats, meta


def normalize_state(state, stats):
    state = np.asarray(state, dtype=np.float32)
    expected = np.asarray(stats["q01"]).shape
    if expected not in ((8,), (10,)) or state.shape != expected or not np.isfinite(state).all():
        raise ValueError(f"模型状态必须按训练定义提供，共 {expected[0]} 个有限值")
    # Same formula/order as the archived OpenPI Normalize._normalize_quantile.
    return (state - stats["q01"]) / (stats["q99"] - stats["q01"] + 1e-6) * 2 - 1


def prompt_tokens(tokenizer, prompt, state, max_len=200):
    clean = prompt.strip().replace("_", " ").replace("\n", " ")
    if not clean:
        raise ValueError("任务描述不能为空")
    bins = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
    text = f"Task: {clean}, State: {' '.join(map(str, bins))};\nAction: "
    ids = tokenizer.encode(text, add_bos=True)
    if len(ids) > max_len:
        raise ValueError(f"任务与状态超过 {max_len} 个 token，请缩短任务描述")
    mask = [True] * len(ids) + [False] * (max_len - len(ids))
    return np.asarray(ids + [0] * (max_len - len(ids))), np.asarray(mask)


def resize_rgb(rgb):
    """Use the same JAX linear resize, uint8 rounding, and black padding as training."""
    import jax
    import jax.numpy as jnp

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("相机图像必须为 HWC uint8 RGB")
    h, w = rgb.shape[:2]
    ratio = max(w / 224, h / 224)
    rh, rw = int(h / ratio), int(w / ratio)
    image = jax.image.resize(jnp.asarray(rgb), (rh, rw, 3), method="linear")
    image = jnp.round(image).clip(0, 255).astype(jnp.uint8)
    ph, pw = (224 - rh) // 2, (224 - rw) // 2
    return np.asarray(jnp.pad(image, ((ph, 224-rh-ph), (pw, 224-rw-pw), (0, 0))))


class Pi05Policy:
    def __init__(self, filename, status=lambda text: None):
        import sentencepiece
        import torch
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
        from transformers.initialization import no_init_weights

        verify_runtime()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，本程序需要 NVIDIA GPU")
        torch.set_num_threads(4)
        status("核对模型文件和归一化参数…")
        self.path, self.stats, self.meta = read_bundle(filename)
        self.contract = self.meta["model_contract"]
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(TOKENIZER))
        status("构建 π0.5 网络并严格加载权重…")
        with no_init_weights():
            self.model = PI05Pytorch(PI05Config(
                dtype="bfloat16", device="cuda", chunk_size=50,
                n_action_steps=50, num_inference_steps=5, compile_model=False,
            ))
        weights = torch.load(self.path, map_location="cpu", mmap=True, weights_only=True)
        # No key rewriting, ignored keys, LoRA conversion, or silently random layers.
        self.model.load_state_dict(weights, strict=True, assign=True)
        del weights
        # LeRobot's current implementation uses an FP32 vision path.
        # Converting these checkpoint values does not alter their stored values.
        self.model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        self.model.eval().requires_grad_(False).to("cuda")
        status("模型已加载；正在预热（不会连接机械臂）…")
        self.predict(
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((480, 640, 3), dtype=np.uint8),
            self.stats["state"]["mean"].astype(np.float32), "Hold position.",
        )

    def policy_tokens(self, prompt, state):
        normalized = normalize_state(state, self.stats["state"])
        start = {"full":0,"rot6d_gripper":3,"gripper":9}[self.meta["policy_state_view"]]
        return prompt_tokens(self.tokenizer, prompt, normalized[start:])

    def predict(self, global_rgb, wrist_rgb, state, prompt, settings=None):
        import torch
        from inference_settings import InferenceSettings
        # Historical diagnostic scripts call predict directly and set torch's
        # seed themselves. Preserve that API; the GUI worker always supplies
        # its explicit per-run settings (2 steps / seed 7 by default).
        options = (InferenceSettings.from_dict(settings) if settings is not None else
                   InferenceSettings(denoise_steps=5, fixed_seed=False))

        start = time.monotonic()
        ids, mask = self.policy_tokens(prompt, state)
        # The third camera slot is present but masked, matching RealworldInputs.
        rgbs = [resize_rgb(global_rgb), resize_rgb(wrist_rgb), np.zeros((224, 224, 3), np.uint8)]
        images = [
            torch.from_numpy(np.array(x, copy=True)).permute(2, 0, 1).unsqueeze(0)
            .to(device="cuda", dtype=torch.float32) / 127.5 - 1 for x in rgbs
        ]
        # Match the staged probe: reset immediately before every prediction.
        if options.fixed_seed:
            torch.manual_seed(options.seed)
        with torch.inference_mode():
            actions = self.model.sample_actions(
                images=images,
                img_masks=[torch.tensor([x], device="cuda") for x in (True, True, False)],
                tokens=torch.as_tensor(ids[None], device="cuda"),
                masks=torch.as_tensor(mask[None], device="cuda"),
                num_steps=options.denoise_steps,
            )[0].float().cpu().numpy()
        s = self.stats["actions"]
        dim = action_dim(self.contract["mode"])
        # Only physical action channels are supervised; accept physical or padded stats.
        actions = actions[:, :dim]
        actions = (actions + 1) / 2 * (s["q99"][:dim] - s["q01"][:dim] + 1e-6) + s["q01"][:dim]
        actions = actions.astype(np.float32)
        if actions.shape != (50, dim) or not np.isfinite(actions).all():
            raise RuntimeError("模型返回的动作形状错误或含 NaN/Inf")
        if self.contract["mode"] == TCP:
            actions = normalize_tcp_actions(actions, state)
        return actions, time.monotonic() - start


def policy_worker(requests, replies):
    """Separate GPU process: a slow inference must never delay pause handling."""
    policy = None
    while True:
        req = requests.get()
        if req["kind"] == "quit":
            return
        token = req["token"]
        try:
            if req["kind"] == "load":
                if policy is not None:
                    del policy
                    policy = None
                    import gc
                    import torch
                    gc.collect()
                    torch.cuda.empty_cache()
                policy = Pi05Policy(req["path"], lambda text: replies.put(
                    {"kind": "progress", "token": token, "text": text}))
                replies.put({"kind": "loaded", "token": token, "path": str(policy.path),
                             "verification": policy.meta, "contract": policy.contract})
            elif req["kind"] == "infer":
                if policy is None:
                    raise RuntimeError("模型尚未加载")
                from inference_settings import InferenceSettings
                actions, elapsed = policy.predict(
                    req["global"], req["wrist"], req["state"], req["prompt"],
                    req.get("settings", InferenceSettings().to_dict()))
                replies.put({"kind": "actions", "token": token, "actions": actions, "elapsed": elapsed})
        except Exception as error:
            import traceback
            replies.put({"kind": "error", "token": token, "text": str(error), "trace": traceback.format_exc()})
