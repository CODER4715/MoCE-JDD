# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from net.moce_jdd import MoCEJDD
from net.moce_jdd_isp import MoCEJDD_ISP
from options import train_options
import yaml
import onnxruntime


def export_model(model, dummy_input, output_path):
    """
    导出模型到 ONNX 格式，并设置动态轴以支持可变输入尺寸。
    """
    # 确保模型处于评估模式
    model.eval()

    # 定义动态轴
    # 我们告诉 ONNX，输入'input'的第0轴(batch), 第3轴(height), 第4轴(width)是可变的
    # 并给它们起了可读的名称 'batch_size', 'height', 'width'
    # 输出'output'的相应维度也会是动态的
    dynamic_axes = {
        'input': {0: 'batch_size', 3: 'height', 4: 'width'},
        'output': {0: 'batch_size', 3: 'height', 4: 'width'}  # 假设输出的高度和宽度也随输入变化
    }

    # 导出模型
    print(f"🚀 Starting ONNX export to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,  # <--- 核心改动在这里
        training=torch.onnx.TrainingMode.EVAL,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        verbose=False,
    )
    print(f"✅ Model exported to {output_path} with dynamic axes.")


def check_determinism(model, dummy_input):
    """
    在导出前测试模型确定性。
    """
    model.eval()
    with torch.no_grad():
        out1 = model(dummy_input)
        out2 = model(dummy_input)
        print(f"🔍 Checking model determinism. Output difference: {torch.max(torch.abs(out1 - out2)).item()}")


def update_model_params(train_opt, yaml_path='hparams.yaml'):
    """只更新模型结构相关参数"""
    MODEL_KEYS = {
        'stage_depth', 'topk', 'num_blocks', 'num_dec_blocks',
        'num_exp_blocks', 'num_refinement_blocks', 'depth_type',
        'dim', 'heads', 'latent_dim', 'complexity_scale'
    }

    if not os.path.exists(yaml_path):
        print(f"⚠️ Warning: hparams.yaml not found at {yaml_path}. Using default parameters.")
        return train_opt

    with open(yaml_path, 'r') as f:
        hparams = yaml.safe_load(f)

    # 只更新模型结构参数
    for k in MODEL_KEYS:
        if k in hparams:
            setattr(train_opt, k, hparams[k])

    print("🔄 Model parameters updated from hparams.yaml.")
    return train_opt

def compare_pytorch_and_onnx(model, onnx_path):
    """
    精确对比 PyTorch 模型和 ONNX 模型的输出。
    """
    print("\n" + "="*50)
    print("🔬 Starting Detailed Comparison: PyTorch vs. ONNX")
    print("="*50)

    # 1. 准备完全相同的输入数据
    # 使用一个固定的、非随机的输入，或者从文件中加载，以保证可复现性
    input_tensor = torch.randn(1, 5, 3, 360, 640, dtype=torch.float32)
    input_numpy = input_tensor.numpy()

    # 2. PyTorch 推理
    model.eval()
    with torch.no_grad():
        pytorch_output = model(input_tensor)
    pytorch_output_np = pytorch_output.detach().numpy()
    print(f"PyTorch output shape: {pytorch_output_np.shape}")

    # 3. ONNX 推理
    ort_session = onnxruntime.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: input_numpy}
    onnx_output_np = ort_session.run(None, ort_inputs)[0]
    print(f"ONNX output shape: {onnx_output_np.shape}")

    # 4. 对比结果
    try:
        np.testing.assert_allclose(pytorch_output_np, onnx_output_np, rtol=1e-3, atol=1e-5)
        print("\n✅ SUCCESS: Outputs are very close!")
    except AssertionError as e:
        print("\n❌ FAILURE: Outputs are significantly different.")
        # 计算并打印差异
        abs_diff = np.abs(pytorch_output_np - onnx_output_np)
        print(f"  - Max absolute difference: {np.max(abs_diff)}")
        print(f"  - Mean absolute difference: {np.mean(abs_diff)}")
        print(f"  - Max relative difference: {np.max(abs_diff / np.abs(pytorch_output_np))}")
def main_export(opt, model_type):
    """
    主函数：加载 PyTorch 模型并导出为 ONNX。
    """
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # 使用 _all 保证多GPU下的一致性

    # 确保使用确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    if model_type!='isp':

        model = MoCEJDD(
            dim=opt.dim,
            num_blocks=opt.num_blocks,
            num_dec_blocks=opt.num_dec_blocks,
            levels=len(opt.num_blocks),
            heads=opt.heads,
            num_refinement_blocks=opt.num_refinement_blocks,
            topk=opt.topk,
            num_experts=opt.num_exp_blocks,
            rank=opt.latent_dim,
            with_complexity=opt.with_complexity,
            depth_type=opt.depth_type,
            stage_depth=opt.stage_depth,
            rank_type=opt.rank_type,
            complexity_scale=opt.complexity_scale,
        )

    else:
        model = MoCEJDD_ISP(
            dim=opt.dim,
            num_blocks=opt.num_blocks,
            num_dec_blocks=opt.num_dec_blocks,
            levels=len(opt.num_blocks),
            heads=opt.heads,
            num_refinement_blocks=opt.num_refinement_blocks,
            topk=opt.topk,
            num_experts=opt.num_exp_blocks,
            rank=opt.latent_dim,
            with_complexity=opt.with_complexity,
            depth_type=opt.depth_type,
            stage_depth=opt.stage_depth,
            rank_type=opt.rank_type,
            complexity_scale=opt.complexity_scale,
        )
    print("🛠️ Model initialized.")

    # --- 加载权重 ---
    if model_type!='isp':
        ckpt_path = os.path.join(opt.ckpt_dir, 'MoCE_JDD', "last.ckpt")
    else:
        ckpt_path = os.path.join(opt.ckpt_dir, 'MoCE_JDD', "last_isp.ckpt")
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Checkpoint file not found at {ckpt_path}")
        print(
            "Please ensure 'options.py' has the correct 'ckpt_dir' and 'checkpoint_id', or provide them as arguments.")
        return  # 找不到权重文件则退出

    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))  # 推荐在CPU上加载以避免GPU内存问题
    model_state_dict = {k.replace('net.', ''): v for k, v in checkpoint['state_dict'].items()
                        if k.startswith('net.')}
    model.load_state_dict(model_state_dict, strict=True)
    print(f"✅ Weights loaded from {ckpt_path}")

    model.eval()

    # --- 创建用于导出的虚拟输入 ---
    # 尺寸： (batch, time, channels, height, width)
    dummy_input = torch.rand(1, 5, 3, 720, 1280, dtype=torch.float32)  # 使用 360p (640x360) 作为导出基准

    # --- 检查与导出 ---
    check_determinism(model, dummy_input)
    export_model(model, dummy_input, f"onnx/model{model_type}.onnx")

    compare_pytorch_and_onnx(model, f"onnx/model{model_type}.onnx")


def test_onnx_model(onnx_path="model.onnx"):
    """
    加载 ONNX 模型并使用 720p 的随机输入进行推理测试。

    Args:
        onnx_path (str): ONNX 模型文件的路径。
    """
    print("\n" + "=" * 50)
    print("⚡ Starting ONNX Runtime Test")
    print("=" * 50)

    if not os.path.exists(onnx_path):
        print(f"❌ Error: ONNX model not found at '{onnx_path}'. Cannot run test.")
        return

    try:
        # 1. 创建 ONNX Runtime 推理会话
        ort_session = onnxruntime.InferenceSession(onnx_path)
        print("✅ ONNX Runtime session created successfully.")

        # 2. 获取模型输入的名称
        input_name = ort_session.get_inputs()[0].name
        print(f"🔍 Model input name: '{input_name}'")

        # 3. 准备一个新的720p测试输入
        # 尺寸：(batch, time, channels, height, width)
        # 720p: 1280x720
        test_input_shape_720p = (1, 5, 3, 720, 1280)
        test_input_720p = np.random.rand(*test_input_shape_720p).astype(np.float32)
        print(f"📦 Prepared a new random input with shape (720p): {test_input_720p.shape}")

        # 4. 执行推理
        print("🚀 Running inference with the new 720p input...")
        ort_inputs = {input_name: test_input_720p}
        ort_outs = ort_session.run(None, ort_inputs)
        output_tensor = ort_outs[0]

        # 5. 打印输出信息
        print("\n🎉 Inference successful!")
        print(f"✅ Output tensor shape: {output_tensor.shape}")
        print(f"✅ Output data type: {output_tensor.dtype}")

    except Exception as e:
        print(f"❌ An error occurred during ONNX Runtime test: {e}")


if __name__ == '__main__':
    model_type = 'isp' # ''/isp
    # 解析命令行参数或使用默认值
    train_opt = train_options()

    # 从 checkpoints/<id>/hparams.yaml 更新模型参数
    if model_type != 'isp':
        hparams_path = os.path.join('checkpoints', train_opt.checkpoint_id, 'hparams.yaml')
    else:
        hparams_path = os.path.join('checkpoints', train_opt.checkpoint_id, 'hparams_isp.yaml')


    train_opt = update_model_params(train_opt, hparams_path)

    # 执行导出
    main_export(train_opt,model_type)

    # --- 步骤 2: 读取导出的 ONNX 模型并进行测试 ---
    test_onnx_model(onnx_path=f"onnx/model{model_type}.onnx")