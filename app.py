#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Amazon Chronos-2 WebUI（hf-mirror：本地权重加载与时间序列预测可视化）

本工程以 hf-mirror 上的 Amazon Chronos-2 为对象，在 template/ 内实现"下载→落盘→本地加载→推理→可视化"闭环。
采用按钮式交互（预设测试时间序列），避免文本框与文件选择，便于 Playwright 自动化截图与操作。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from chronos import Chronos2Pipeline
    ChronosPipeline = Chronos2Pipeline  # 使用 Chronos2Pipeline
except ImportError:
    from chronos import ChronosPipeline  # 降级到 ChronosPipeline

matplotlib.use("Agg")

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
REPO_ID = "amazon/chronos-2"
REVISION = os.environ.get("REVISION", "main")

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models" / "amazon--chronos-2"
ASSETS_DIR = PROJECT_DIR / "assets"
SCREENSHOT_DIR = PROJECT_DIR / "screenshots"
os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)

_pipeline: Optional[ChronosPipeline] = None

# 预设测试时间序列（生成示例数据，避免文件上传）
def generate_test_series(series_type: str = "trend") -> pd.Series:
    """生成测试用的时间序列数据"""
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start="2020-01-01", periods=n, freq="D")
    
    if series_type == "trend":
        # 趋势序列
        values = np.linspace(10, 50, n) + np.random.normal(0, 2, n)
    elif series_type == "seasonal":
        # 季节性序列
        t = np.arange(n)
        values = 30 + 10 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 2, n)
    elif series_type == "random":
        # 随机游走
        values = np.cumsum(np.random.normal(0, 1, n)) + 20
    else:  # mixed
        # 混合模式
        t = np.arange(n)
        trend = np.linspace(20, 40, n)
        seasonal = 5 * np.sin(2 * np.pi * t / 30)
        noise = np.random.normal(0, 1.5, n)
        values = trend + seasonal + noise
    
    return pd.Series(values, index=dates, name="value")


PRESET_TESTS: List[Tuple[str, str]] = [
    ("测试 1：趋势序列", "trend"),
    ("测试 2：季节性序列", "seasonal"),
    ("测试 3：随机游走", "random"),
    ("测试 4：混合模式", "mixed"),
]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_original_readme() -> str:
    """保存原始模型卡 README 到 assets。"""
    _ensure_dir(ASSETS_DIR)
    try:
        from huggingface_hub import hf_hub_download
        
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename="README.md",
            revision=REVISION,
            endpoint=HF_ENDPOINT,
        )
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        out = ASSETS_DIR / "original_amazon_chronos-2_README.md"
        out.write_text(text, encoding="utf-8")
        return f"已保存原始 README 至：{out}"
    except Exception as e:
        return f"保存原始 README 失败：{e}"


def download_model() -> str:
    """触发下载并落盘到 template/models/amazon--chronos-2/。"""
    from huggingface_hub import snapshot_download

    _ensure_dir(MODEL_DIR)
    try:
        snapshot_download(
            repo_id=REPO_ID,
            revision=REVISION,
            local_dir=str(MODEL_DIR),
            endpoint=HF_ENDPOINT,
        )
        # 检查模型文件
        model_files = list(MODEL_DIR.glob("*.safetensors")) + list(MODEL_DIR.glob("*.bin"))
        size_mb = sum(p.stat().st_size for p in model_files) / 1024 / 1024 if model_files else 0
        return (
            "下载完成。\n"
            f"- HF_ENDPOINT: {HF_ENDPOINT}\n"
            f"- 本地目录: {MODEL_DIR}\n"
            f"- 权重约: {size_mb:.2f} MiB\n"
            "请点击「加载模型」后使用「一键测试」按钮。"
        )
    except Exception as e:
        return f"下载失败：{e}"


def load_model() -> str:
    """从本地目录加载模型（使用已下载的缓存）。"""
    global _pipeline

    if not MODEL_DIR.exists():
        return f"本地目录不存在：{MODEL_DIR}。请先点击「下载模型」。"
    
    if _pipeline is not None:
        return "模型已加载，可直接使用「一键测试」按钮。"

    t0 = time.time()
    try:
        # 由于 chronos-forecasting 包版本与模型配置可能存在兼容性问题
        # 我们使用模型ID加载，HuggingFace Hub 会自动使用本地缓存
        # 如果遇到配置问题，会尝试从 transformers 加载后手动创建 pipeline
        try:
            _pipeline = ChronosPipeline.from_pretrained(
                REPO_ID,
                endpoint=HF_ENDPOINT,
            )
        except TypeError as e:
            if 'input_patch_size' in str(e):
                # 如果遇到 input_patch_size 错误，尝试修复配置后重新加载
                import json
                import shutil
                config_path = MODEL_DIR / "config.json"
                if config_path.exists():
                    # 备份原配置
                    backup_path = MODEL_DIR / "config.json.backup"
                    if not backup_path.exists():
                        shutil.copy(config_path, backup_path)
                    
                    # 读取并修改配置
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    if 'chronos_config' in config and isinstance(config['chronos_config'], dict):
                        chronos_config = config['chronos_config'].copy()
                        # 移除不支持的参数
                        chronos_config.pop('input_patch_size', None)
                        config['chronos_config'] = chronos_config
                        
                        # 保存修改后的配置
                        with open(config_path, 'w', encoding='utf-8') as f:
                            json.dump(config, f, indent=2, ensure_ascii=False)
                    
                    # 重新尝试加载
                    _pipeline = ChronosPipeline.from_pretrained(
                        REPO_ID,
                        endpoint=HF_ENDPOINT,
                    )
                else:
                    raise
            else:
                raise
        
        dt = time.time() - t0
        return f"加载成功（使用本地缓存）。\n- 模型: {REPO_ID}\n- 缓存路径: {MODEL_DIR}\n- 耗时: {dt:.2f}s"
    except Exception as e:
        _pipeline = None
        import traceback
        error_msg = f"加载失败：{e}\n\n{traceback.format_exc()}\n提示：如果模型未下载，请先点击「下载模型」按钮。"
        return error_msg


def _infer(series_type: str) -> Tuple[str, str, Any]:
    """执行单次推理，返回 (状态消息, 输出文本, 可视化图)。"""
    global _pipeline

    if _pipeline is None:
        return "请先点击「加载模型」。", "", None

    try:
        # 生成测试数据
        series = generate_test_series(series_type)
        
        # 执行预测
        t0 = time.time()
        forecast = _pipeline.predict(
            series,
            prediction_length=20,  # 预测未来20个时间点
        )
        dt = time.time() - t0

        # 准备可视化
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制历史数据
        ax.plot(series.index, series.values, label="历史数据", color="#4C78A8", linewidth=2)
        
        # 绘制预测结果
        if isinstance(forecast, pd.Series):
            forecast_index = pd.date_range(
                start=series.index[-1] + pd.Timedelta(days=1),
                periods=len(forecast),
                freq="D"
            )
            ax.plot(forecast_index, forecast.values, label="预测结果", color="#E45756", linewidth=2, linestyle="--")
        elif isinstance(forecast, pd.DataFrame):
            # 如果有多个分位数，绘制中位数和置信区间
            if "0.5" in forecast.columns:
                forecast_index = pd.date_range(
                    start=series.index[-1] + pd.Timedelta(days=1),
                    periods=len(forecast),
                    freq="D"
                )
                ax.plot(forecast_index, forecast["0.5"], label="预测中位数", color="#E45756", linewidth=2, linestyle="--")
                if "0.1" in forecast.columns and "0.9" in forecast.columns:
                    ax.fill_between(
                        forecast_index,
                        forecast["0.1"],
                        forecast["0.9"],
                        alpha=0.3,
                        color="#E45756",
                        label="80% 置信区间"
                    )
            else:
                # 使用第一列
                forecast_index = pd.date_range(
                    start=series.index[-1] + pd.Timedelta(days=1),
                    periods=len(forecast),
                    freq="D"
                )
                ax.plot(forecast_index, forecast.iloc[:, 0], label="预测结果", color="#E45756", linewidth=2, linestyle="--")
        else:
            # 转换为 Series
            forecast_index = pd.date_range(
                start=series.index[-1] + pd.Timedelta(days=1),
                periods=len(forecast),
                freq="D"
            )
            forecast_series = pd.Series(forecast.flatten() if hasattr(forecast, 'flatten') else forecast)
            ax.plot(forecast_index, forecast_series.values, label="预测结果", color="#E45756", linewidth=2, linestyle="--")
        
        # 添加分界线
        ax.axvline(x=series.index[-1], color="gray", linestyle=":", linewidth=1, label="预测起点")
        
        ax.set_xlabel("日期", fontsize=12)
        ax.set_ylabel("数值", fontsize=12)
        ax.set_title(f"Chronos-2 时间序列预测 ({series_type})", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        # 生成统计信息
        summary = (
            f"测试类型: {series_type}\n\n"
            f"历史数据点数: {len(series)}\n"
            f"预测长度: {len(forecast) if hasattr(forecast, '__len__') else 'N/A'}\n"
            f"预测耗时: {dt:.3f}s\n\n"
            f"历史数据范围: [{series.min():.2f}, {series.max():.2f}]\n"
        )
        if isinstance(forecast, pd.Series):
            summary += f"预测范围: [{forecast.min():.2f}, {forecast.max():.2f}]"
        elif isinstance(forecast, pd.DataFrame) and "0.5" in forecast.columns:
            summary += f"预测中位数范围: [{forecast['0.5'].min():.2f}, {forecast['0.5'].max():.2f}]"
        
        return "推理完成。", summary, fig
    except Exception as e:
        import traceback
        error_msg = f"推理失败：{e}\n\n{traceback.format_exc()}"
        return error_msg, "", None


def make_infer_fn(series_type: str):
    def fn():
        return _infer(series_type)
    return fn


_ensure_dir(ASSETS_DIR)
_ensure_dir(SCREENSHOT_DIR)

PORT = int(os.environ.get("PORT", "7890"))

with gr.Blocks(title="Chronos-2 WebUI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Amazon Chronos-2 WebUI（hf-mirror 本地权重加载与时间序列预测可视化）")
    gr.Markdown(
        "本界面以 Amazon/Chronos-2 为中心，在 template/ 内完成「下载→加载→推理→可视化」闭环。"
        "采用按钮式交互与预设测试时间序列，便于自动化截图与复现。"
    )

    with gr.Row():
        btn_save = gr.Button("🧾 保存原始模型卡（README）", variant="secondary")
        btn_dl = gr.Button("⬇️ 下载模型到本地", variant="primary")
        btn_load = gr.Button("📦 加载模型（仅本地）", variant="secondary")
    status = gr.Textbox(label="状态", value="未开始", interactive=False, lines=6)

    btn_save.click(save_original_readme, outputs=status)
    btn_dl.click(download_model, outputs=status)
    btn_load.click(load_model, outputs=status)

    gr.Markdown("## 一键测试（预设时间序列，按钮操作）")
    btns = []
    with gr.Row():
        for label, _ in PRESET_TESTS:
            btns.append(gr.Button(label, variant="secondary"))

    out_text = gr.Textbox(label="输出（推理结果与统计）", lines=12, interactive=False)
    out_plot = gr.Plot(label="时间序列预测可视化")

    for i, (label, series_type) in enumerate(PRESET_TESTS):
        btns[i].click(
            make_infer_fn(series_type),
            outputs=[status, out_text, out_plot],
        )

    gr.Markdown(
        "说明：点击上方任一测试按钮将使用对应预设时间序列进行预测，结果与可视化图表显示于下方。"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT, share=False)
