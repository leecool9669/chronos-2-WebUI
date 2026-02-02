import math
import random
from datetime import datetime
from typing import List, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np


def _parse_series(raw: str) -> List[float]:
    """Parse a comma/space separated numeric sequence."""
    tokens = [t.strip() for t in raw.replace("\n", " ").split(",")]
    values: List[float] = []
    for tok in tokens:
        if not tok:
            continue
        try:
            values.append(float(tok))
        except ValueError:
            # Ignore non-numeric tokens to keep the demo robust.
            continue
    return values


def _simulate_forecast(
    history: List[float],
    prediction_length: int,
    quantiles: Tuple[float, float, float] = (0.1, 0.5, 0.9),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple statistical toy forecaster used purely for UI demo.

    It computes a rolling mean and adds Gaussian noise; the interface
    mimics probabilistic forecasting with three quantiles.
    """
    if not history:
        base = np.zeros(prediction_length)
    else:
        hist = np.asarray(history, dtype=float)
        window = min(len(hist), max(4, len(hist) // 4))
        mean = np.convolve(hist, np.ones(window) / window, mode="valid")[-1]
        trend = (hist[-1] - hist[0]) / max(1, len(hist) - 1)
        base = mean + trend * np.arange(1, prediction_length + 1)

    noise_scale = max(1e-3, float(np.std(history) if history else 1.0) * 0.15)
    eps = np.random.normal(0.0, noise_scale, size=(prediction_length,))

    q10, q50, q90 = quantiles
    median = base + eps
    lower = median - abs(q50 - q10) * noise_scale * 2.0
    upper = median + abs(q90 - q50) * noise_scale * 2.0

    return base, lower, median, upper


def forecast_interface(
    raw_series: str,
    prediction_length: int,
    seed: int,
    task_type: str,
    covariates: str,
    model_variant: str,
):
    """
    Core callback used by Gradio.

    This function does NOT call the real Chronos-2 model; instead it
    performs a lightweight simulation so that the UI can be tested
    without downloading any weights.
    """
    random.seed(seed)
    np.random.seed(seed)

    history = _parse_series(raw_series)
    if len(history) < 4:
        return (
            "请输入至少 4 个历史观测值，以英文逗号分隔，例如：12.3, 13.1, 12.8, 14.0, 13.7",
            None,
        )

    base, lower, median, upper = _simulate_forecast(history, prediction_length)

    # Build a matplotlib figure for visualization.
    fig, ax = plt.subplots(figsize=(7, 4))
    time_hist = np.arange(len(history))
    time_pred = np.arange(len(history), len(history) + prediction_length)

    ax.plot(time_hist, history, label="历史观测", color="#1f77b4")
    ax.plot(time_pred, median, label="预测中位数", color="#ff7f0e")
    ax.fill_between(
        time_pred,
        lower,
        upper,
        color="#ffbb78",
        alpha=0.3,
        label="预测区间 (10%–90%)",
    )

    ax.set_xlabel("时间步")
    ax.set_ylabel("数值")
    ax.set_title(f"Chronos-2 模拟预测结果（变体：{model_variant}，任务：{task_type}）")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    description = (
        f"本界面模拟了 Chronos-2 在时间序列上的概率预测流程："
        f"给定 {len(history)} 个历史观测点，在任务类型“{task_type}”与协变量设定“{covariates or '无'}”下，"
        f"输出长度为 {prediction_length} 的预测序列，并以区间形式呈现不确定性。"
        "当前实现仅为前端演示，不会触发任何真实模型或大文件下载，"
        "适合作为后续接入 chronos-forecasting 推理管线的工程样板。"
    )

    return description, fig


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Chronos-2 WebUI Demo") as demo:
        gr.Markdown(
            """
        # Chronos-2 时间序列预测 WebUI（演示版）

        本界面围绕时间序列基础模型 **Chronos-2** 设计，仅提供交互与可视化流程的演示。
        为避免下载大规模权重文件，本示例中的“模型推理”通过轻量级统计模拟完成，
        但接口签名尽量贴近真实的 `chronos-forecasting` 推理管线，便于后续替换为真是模型。
        """
        )

        with gr.Row():
            with gr.Column(scale=2):
                series_input = gr.Textbox(
                    label="历史时间序列（以英文逗号分隔）",
                    placeholder="例如：12.3, 13.1, 12.8, 14.0, 13.7, 15.2, ...",
                    lines=5,
                )
                covariates = gr.Textbox(
                    label="协变量描述（可选，自然语言）",
                    placeholder="例如：节假日标记、价格区间、气候指标等，仅用于说明，不参与模拟计算。",
                )

            with gr.Column(scale=1):
                model_variant = gr.Dropdown(
                    choices=[
                        "chronos-2 (base, 120M)",
                        "chronos-2-ts (time-series optimized)",
                        "chronos-2-lite (示意用轻量版)",
                    ],
                    value="chronos-2 (base, 120M)",
                    label="模型变体（示意）",
                )
                task_type = gr.Radio(
                    label="任务类型",
                    choices=["单变量预测", "多变量联合预测", "协变量增强预测"],
                    value="单变量预测",
                )
                prediction_length = gr.Slider(
                    minimum=4,
                    maximum=96,
                    step=4,
                    value=24,
                    label="预测长度（时间步）",
                )
                seed = gr.Slider(
                    minimum=0,
                    maximum=10_000,
                    step=1,
                    value=2026,
                    label="随机种子（用于模拟可复现）",
                )
                run_btn = gr.Button("生成模拟预测", variant="primary")

        description = gr.Markdown(label="文字说明")
        plot = gr.Plot(label="预测结果可视化")

        run_btn.click(
            forecast_interface,
            inputs=[
                series_input,
                prediction_length,
                seed,
                task_type,
                covariates,
                model_variant,
            ],
            outputs=[description, plot],
        )

        gr.Markdown(
            f"界面构建时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}（仅作展示用时间戳）"
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    # share=False 避免外网暴露，适合作为本地演示。
    demo.launch()
