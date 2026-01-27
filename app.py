"""
Chronos-2 WebUI - 时间序列预测模型可视化演示界面
基于 Gradio 构建，用于展示 Chronos-2 模型的时间序列预测功能
"""

import gradio as gr
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端


def fake_load_model() -> str:
    """模拟模型加载过程"""
    return """✓ 模型已加载 (演示模式)
模型: autogluon/chronos-2
参数量: 120M
架构: T5 Encoder-based
上下文长度: 8192
最大预测长度: 1024
支持任务: 单变量、多变量、协变量预测
设备: CPU/GPU
状态: 就绪"""


def generate_sample_timeseries(length: int = 100) -> List[float]:
    """生成示例时间序列数据"""
    np.random.seed(42)
    trend = np.linspace(0, 10, length)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(length) / 12)
    noise = np.random.normal(0, 1, length)
    return (trend + seasonal + noise).tolist()


def fake_predict(
    timeseries_data: str,
    prediction_length: int = 24,
    quantile_levels: str = "0.1,0.5,0.9",
    use_covariates: bool = False
) -> Tuple[str, str]:
    """
    模拟时间序列预测过程
    
    Args:
        timeseries_data: 时间序列数据（JSON格式或逗号分隔）
        prediction_length: 预测长度
        quantile_levels: 分位数水平
        use_covariates: 是否使用协变量
    
    Returns:
        (预测结果文本, 可视化说明)
    """
    if not timeseries_data or not timeseries_data.strip():
        return "请先输入时间序列数据", "等待数据输入..."
    
    try:
        # 解析输入数据
        if timeseries_data.strip().startswith('[') or timeseries_data.strip().startswith('{'):
            data = json.loads(timeseries_data)
            if isinstance(data, dict):
                values = data.get('values', data.get('target', []))
            else:
                values = data
        else:
            # 逗号分隔的数值
            values = [float(x.strip()) for x in timeseries_data.split(',') if x.strip()]
        
        if not values:
            return "无法解析时间序列数据", "数据格式错误"
        
        # 生成模拟预测结果
        last_value = values[-1]
        trend = np.linspace(0, 2, prediction_length)
        seasonal = 0.5 * np.sin(2 * np.pi * np.arange(prediction_length) / 12)
        noise = np.random.normal(0, 0.3, prediction_length)
        
        # 中位数预测
        median_forecast = (last_value + trend + seasonal + noise).tolist()
        
        # 分位数预测
        quantiles = [float(q.strip()) for q in quantile_levels.split(',')]
        quantiles.sort()
        
        result_text = f"""
**时间序列预测结果:**

**输入数据:**
- 历史数据长度: {len(values)} 个时间点
- 最后观测值: {values[-1]:.4f}
- 数据范围: [{min(values):.4f}, {max(values):.4f}]

**预测参数:**
- 预测长度: {prediction_length} 个时间步
- 分位数水平: {', '.join([f'{q:.2f}' for q in quantiles])}
- 使用协变量: {'是' if use_covariates else '否'}

**预测结果 (中位数):**
- 预测值范围: [{min(median_forecast):.4f}, {max(median_forecast):.4f}]
- 平均预测值: {np.mean(median_forecast):.4f}
- 预测趋势: {'上升' if median_forecast[-1] > median_forecast[0] else '下降'}

**分位数预测区间:**
"""
        for q in quantiles:
            if q == 0.5:
                continue
            q_forecast = [v + (q - 0.5) * 2 for v in median_forecast]
            result_text += f"- {q*100:.0f}% 分位数: [{min(q_forecast):.4f}, {max(q_forecast):.4f}]\n"
        
        viz_text = f"""
## 时间序列预测可视化

**模型处理流程:**

1. **数据预处理:**
   - 输入序列长度: {len(values)} 个时间点
   - 上下文窗口: 8192 (最大)
   - 数据归一化: 使用 arcsinh 变换
   - 时间编码: 位置编码 + 时间特征

2. **模型推理:**
   - 架构: T5 Encoder-based (12层, 12个注意力头)
   - 输入补丁大小: 16
   - 输出补丁大小: 16
   - 组注意力机制: 跨序列和协变量的上下文学习

3. **预测生成:**
   - 多步预测: 生成 {prediction_length} 个未来时间步
   - 分位数预测: 生成 {len(quantiles)} 个分位数水平
   - 不确定性量化: 通过分位数区间表示预测不确定性

**模型特点:**
- **零样本预测**: 无需针对特定数据集微调
- **多任务支持**: 单变量、多变量、协变量预测统一架构
- **高效推理**: 单GPU上每秒可处理300+时间序列
- **长上下文**: 支持最长8192的上下文窗口
- **长预测**: 支持最长1024步的预测长度

**技术原理:**
Chronos-2 基于 T5 编码器架构，采用补丁化（patch-based）方法处理时间序列。
模型将时间序列分割成固定大小的补丁，通过 Transformer 编码器学习时间序列的表示。
使用组注意力机制实现跨序列和协变量的高效上下文学习，支持零样本预测。

**应用场景:**
- 金融预测: 股票价格、汇率预测
- 需求预测: 销售、库存预测
- 能源预测: 电力负荷、能源消耗预测
- 交通预测: 交通流量、出行需求预测
- 气象预测: 温度、降水量预测
"""
        
        return result_text, viz_text
    
    except Exception as e:
        return f"处理错误: {str(e)}", "数据解析失败"


def create_visualization(
    historical: List[float],
    forecast: List[float],
    quantiles: List[float]
) -> str:
    """创建预测结果可视化图表"""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 历史数据
        hist_len = len(historical)
        hist_x = np.arange(hist_len)
        ax.plot(hist_x, historical, 'b-', label='历史数据', linewidth=2)
        
        # 预测数据
        forecast_x = np.arange(hist_len, hist_len + len(forecast))
        ax.plot(forecast_x, forecast, 'r--', label='预测值 (中位数)', linewidth=2)
        
        # 分位数区间
        if quantiles:
            lower = [v - 1.0 for v in forecast]
            upper = [v + 1.0 for v in forecast]
            ax.fill_between(forecast_x, lower, upper, alpha=0.3, color='red', label='预测区间')
        
        ax.axvline(x=hist_len - 0.5, color='gray', linestyle='--', linewidth=1, label='预测起点')
        ax.set_xlabel('时间步', fontsize=12)
        ax.set_ylabel('数值', fontsize=12)
        ax.set_title('Chronos-2 时间序列预测结果', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('prediction_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return 'prediction_plot.png'
    except Exception as e:
        return None


def build_ui():
    """构建 Gradio WebUI 界面"""
    with gr.Blocks(title="Chronos-2 WebUI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # Chronos-2 时间序列预测模型 · WebUI 演示
        
        这是一个基于 Gradio 的 Chronos-2 模型可视化界面，用于测试和演示时间序列预测功能。
        
        **模型信息:**
        - 模型名称: autogluon/chronos-2
        - 参数量: 120M
        - 架构: T5 Encoder-based
        - 上下文长度: 8192
        - 最大预测长度: 1024
        - 支持任务: 单变量、多变量、协变量预测
        """)
        
        # 模型加载区
        with gr.Row():
            load_btn = gr.Button("加载模型（演示）", variant="primary", size="lg")
            status_box = gr.Textbox(
                label="模型状态",
                value="尚未加载",
                interactive=False,
                lines=6
            )
        load_btn.click(fn=fake_load_model, outputs=status_box)
        
        gr.Markdown("---")
        
        # 主要功能区域
        with gr.Tabs():
            # 单变量预测标签页
            with gr.Tab("单变量预测"):
                gr.Markdown("""
                ### 单变量时间序列预测
                
                输入历史时间序列数据，模型将生成未来时间步的预测值。支持分位数预测以量化不确定性。
                """)
                
                with gr.Row():
                    with gr.Column():
                        timeseries_input = gr.Textbox(
                            label="时间序列数据",
                            placeholder='输入格式: [1.2, 3.4, 5.6, ...] 或 1.2,3.4,5.6,...',
                            lines=8,
                            value=""
                        )
                        sample_btn = gr.Button("生成示例数据", variant="secondary", size="sm")
                        
                        with gr.Row():
                            prediction_length = gr.Slider(
                                label="预测长度",
                                minimum=1,
                                maximum=1024,
                                value=24,
                                step=1,
                                info="预测未来多少个时间步"
                            )
                            quantile_levels = gr.Textbox(
                                label="分位数水平",
                                value="0.1,0.5,0.9",
                                info="逗号分隔，如: 0.1,0.5,0.9"
                            )
                        
                        use_covariates = gr.Checkbox(
                            label="使用协变量",
                            value=False,
                            info="是否使用外部协变量进行预测"
                        )
                        
                        predict_btn = gr.Button("开始预测", variant="primary")
                    
                    with gr.Column():
                        prediction_output = gr.Markdown(
                            label="预测结果",
                            value="等待预测..."
                        )
                        visualization_output = gr.Markdown(
                            label="处理过程可视化",
                            value="等待处理..."
                        )
                
                def load_sample():
                    sample_data = generate_sample_timeseries(100)
                    return json.dumps(sample_data)
                
                sample_btn.click(fn=load_sample, outputs=timeseries_input)
                
                predict_btn.click(
                    fn=fake_predict,
                    inputs=[timeseries_input, prediction_length, quantile_levels, use_covariates],
                    outputs=[prediction_output, visualization_output]
                )
            
            # 多变量预测标签页
            with gr.Tab("多变量预测"):
                gr.Markdown("""
                ### 多变量时间序列预测
                
                输入多个相关的时间序列，模型将同时预测它们的未来值。支持跨序列学习。
                """)
                
                multivariate_input = gr.Textbox(
                    label="多变量时间序列数据 (JSON格式)",
                    placeholder='{"series1": [1,2,3,...], "series2": [4,5,6,...]}',
                    lines=10
                )
                multivariate_output = gr.Markdown(
                    label="多变量预测结果",
                    value="等待预测..."
                )
                multivariate_btn = gr.Button("多变量预测", variant="primary")
                
                multivariate_btn.click(
                    fn=lambda x: ("多变量预测功能演示\n\nChronos-2 支持同时预测多个相关时间序列，通过组注意力机制实现跨序列学习。", 
                                 "多变量预测允许模型利用不同序列之间的相关性，提高预测准确性。"),
                    inputs=multivariate_input,
                    outputs=[multivariate_output, gr.Textbox(visible=False)]
                )
            
            # 模型信息标签页
            with gr.Tab("模型信息"):
                gr.Markdown("""
                ## Chronos-2 模型详细信息
                
                ### 模型架构
                
                Chronos-2 基于 T5 编码器架构构建，专门设计用于时间序列预测：
                - **编码器层数**: 12层
                - **注意力头数**: 12个
                - **隐藏层大小**: 768
                - **前馈网络大小**: 3072
                - **输入补丁大小**: 16
                - **输出补丁大小**: 16
                - **最大上下文长度**: 8192
                - **最大预测长度**: 1024
                
                ### 核心特性
                
                1. **零样本预测**: 无需针对特定数据集微调即可进行预测
                2. **多任务支持**: 统一架构支持单变量、多变量和协变量预测
                3. **分位数预测**: 生成多个分位数水平，量化预测不确定性
                4. **高效推理**: 单GPU上每秒可处理300+时间序列
                5. **长上下文**: 支持最长8192的上下文窗口
                6. **长预测**: 支持最长1024步的预测长度
                
                ### 技术原理
                
                Chronos-2 采用补丁化（patch-based）方法处理时间序列：
                1. 将时间序列分割成固定大小的补丁（patch）
                2. 通过 Transformer 编码器学习时间序列的表示
                3. 使用组注意力机制实现跨序列和协变量的上下文学习
                4. 生成多步预测和分位数预测
                
                ### 训练数据
                
                模型在以下数据集上训练：
                - Chronos Datasets 子集
                - GIFT-Eval Pretrain 子集
                - 合成单变量和多变量数据
                
                ### 性能表现
                
                在多个基准测试中达到最先进的零样本准确率：
                - **fev-bench**: 领先的零样本性能
                - **GIFT-Eval**: 优异的泛化能力
                - **Chronos Benchmark II**: 最佳表现
                
                ### 使用场景
                
                - **金融预测**: 股票价格、汇率、利率预测
                - **需求预测**: 销售预测、库存管理
                - **能源预测**: 电力负荷、能源消耗预测
                - **交通预测**: 交通流量、出行需求预测
                - **气象预测**: 温度、降水量、风速预测
                - **供应链**: 需求预测、库存优化
                
                ### 技术优势
                
                1. **通用性强**: 适用于多种时间序列预测任务
                2. **零样本能力**: 无需微调即可使用
                3. **高效推理**: 支持大规模批量预测
                4. **不确定性量化**: 通过分位数预测提供预测区间
                5. **灵活输入**: 支持单变量、多变量和协变量
                """)
        
        gr.Markdown("---")
        gr.Markdown("""
        **注意**: 当前版本为演示模式，不会实际加载模型或进行真实推理。
        所有结果均为模拟输出，用于展示界面交互流程。
        """)
    
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False
    )
