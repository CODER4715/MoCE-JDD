import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnxruntime


class HamiltonAdamONNXDynamic(nn.Module):
    """
    Hamilton-Adams demosaicing algorithm, modified to support DYNAMIC input sizes
    for ONNX export.

    The key change is that masks are no longer pre-computed and stored as static
    buffers. Instead, they are generated dynamically within the `forward` method
    based on the shape of the incoming tensor. This allows the exported ONNX
    model to handle images of any resolution.
    """

    def __init__(self, pattern: str):
        super().__init__()
        if pattern not in ['rggb', 'grbg', 'gbrg', 'bggr']:
            raise ValueError(f"Unsupported Bayer pattern: {pattern}")
        self.pattern = pattern
        self._init_and_register_kernels()

    def _init_and_register_kernels(self):
        """Initializes all convolution kernels and registers them as non-trainable buffers."""
        k1 = torch.zeros((6, 1, 5, 5), dtype=torch.float32)
        k1[0, 0, 2, 1], k1[0, 0, 2, 3] = 0.5, 0.5  # Kh
        k1[1, 0, 1, 2], k1[1, 0, 3, 2] = 0.5, 0.5  # Kv
        k1[2, 0, 2, 0], k1[2, 0, 2, 2], k1[2, 0, 2, 4] = 1.0, -2.0, 1.0  # Deltah
        k1[3, 0, 0, 2], k1[3, 0, 2, 2], k1[3, 0, 4, 2] = 1.0, -2.0, 1.0  # Deltav
        k1[4, 0, 2, 1], k1[4, 0, 2, 3] = 1.0, -1.0  # Diffh
        k1[5, 0, 1, 2], k1[5, 0, 3, 2] = 1.0, -1.0  # Diffv
        self.register_buffer('kernels_algo1', k1)

        kc = torch.zeros((6, 1, 3, 3), dtype=torch.float32)
        kg = torch.zeros((4, 1, 3, 3), dtype=torch.float32)
        kc[0, 0, 1, 0], kc[0, 0, 1, 2] = 0.5, 0.5  # Kh
        kc[1, 0, 0, 1], kc[1, 0, 2, 1] = 0.5, 0.5  # Kv
        kc[2, 0, 0, 0], kc[2, 0, 2, 2] = 0.5, 0.5  # Kp
        kc[3, 0, 0, 2], kc[3, 0, 2, 0] = 0.5, 0.5  # Kn
        kc[4, 0, 0, 0], kc[4, 0, 2, 2] = -1.0, 1.0  # Diffp
        kc[5, 0, 0, 2], kc[5, 0, 2, 0] = -1.0, 1.0  # Diffn
        kg[0, 0, 1, 0], kg[0, 0, 1, 1], kg[0, 0, 1, 2] = 0.25, -0.5, 0.25  # Deltah
        kg[1, 0, 0, 1], kg[1, 0, 1, 1], kg[1, 0, 2, 1] = 0.25, -0.5, 0.25  # Deltav
        kg[2, 0, 0, 0], kg[2, 0, 1, 1], kg[2, 0, 2, 2] = 1.0, -2.0, 1.0  # Deltap
        kg[3, 0, 0, 2], kg[3, 0, 1, 1], kg[3, 0, 2, 0] = 1.0, -2.0, 1.0  # Deltan
        self.register_buffer('kernels_algo2_chan', kc)
        self.register_buffer('kernels_algo2_green', kg)

    def _generate_masks(self, H, W, device):
        """Dynamically generates masks based on input dimensions."""
        c_map = {'r': 0, 'g': 1, 'b': 2}
        mosaic_mask = torch.zeros((3, H, W), dtype=torch.float32, device=device)
        mosaic_mask[c_map[self.pattern[0]], 0::2, 0::2] = 1
        mosaic_mask[c_map[self.pattern[1]], 0::2, 1::2] = 1
        mosaic_mask[c_map[self.pattern[2]], 1::2, 0::2] = 1
        mosaic_mask[c_map[self.pattern[3]], 1::2, 1::2] = 1

        maskGr = torch.zeros((H, W), dtype=torch.float32, device=device)
        maskGb = torch.zeros((H, W), dtype=torch.float32, device=device)
        if self.pattern == 'grbg':
            maskGr[0::2, 0::2], maskGb[1::2, 1::2] = 1, 1
        elif self.pattern == 'rggb':
            maskGr[0::2, 1::2], maskGb[1::2, 0::2] = 1, 1
        elif self.pattern == 'gbrg':
            maskGb[0::2, 0::2], maskGr[1::2, 1::2] = 1, 1
        elif self.pattern == 'bggr':
            maskGb[0::2, 1::2], maskGr[1::2, 0::2] = 1, 1

        return mosaic_mask, maskGr, maskGb

    def _pack_in_one(self, x, out_H, out_W):
        """Packs 4 bayer channels into a single 2D image."""
        B = x.shape[0]
        y = torch.zeros((B, out_H, out_W), dtype=x.dtype, device=x.device)
        y[:, 0::2, 0::2] = x[:, 0]
        y[:, 0::2, 1::2] = x[:, 1]
        y[:, 1::2, 0::2] = x[:, 2]
        y[:, 1::2, 1::2] = x[:, 3]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation graph for demosaicing."""
        _B, _C, H, W = x.shape
        out_H, out_W = 2 * H, 2 * W

        # Dynamically generate masks based on input shape
        mosaic_mask, mask_Gr, mask_Gb = self._generate_masks(out_H, out_W, x.device)

        x_packed = self._pack_in_one(x, out_H, out_W)
        x_masked = x_packed.unsqueeze(1) * mosaic_mask.unsqueeze(0)
        rawq = torch.sum(x_masked, axis=1)

        conv_rawq = F.conv2d(rawq.unsqueeze(1), self.kernels_algo1, padding=(2, 2))
        rawh = conv_rawq[:, 0] - conv_rawq[:, 2] / 4
        rawv = conv_rawq[:, 1] - conv_rawq[:, 3] / 4
        CLh = torch.abs(conv_rawq[:, 4]) + torch.abs(conv_rawq[:, 2])
        CLv = torch.abs(conv_rawq[:, 5]) + torch.abs(conv_rawq[:, 3])
        CLlocation = torch.sign(CLh - CLv)
        green_interp = (1 + CLlocation) * rawv / 2 + (1 - CLlocation) * rawh / 2
        green = green_interp * (1 - mosaic_mask[1]) + rawq * mosaic_mask[1]
        green = green.unsqueeze(1)

        conv_r = F.conv2d(x_masked[:, 0].unsqueeze(1), self.kernels_algo2_chan, padding=(1, 1))
        conv_g_for_r = F.conv2d(green, self.kernels_algo2_green, padding=(1, 1))
        Ch_r = mask_Gr * (conv_r[:, 0] - conv_g_for_r[:, 0])
        Cv_r = mask_Gb * (conv_r[:, 1] - conv_g_for_r[:, 1])
        Cp_r = mosaic_mask[2] * (conv_r[:, 2] - conv_g_for_r[:, 2] / 4)
        Cn_r = mosaic_mask[2] * (conv_r[:, 3] - conv_g_for_r[:, 3] / 4)
        CLp_r = mosaic_mask[2] * (torch.abs(conv_r[:, 4]) + torch.abs(conv_g_for_r[:, 2]))
        CLn_r = mosaic_mask[2] * (torch.abs(conv_r[:, 5]) + torch.abs(conv_g_for_r[:, 3]))
        CLloc_r = torch.sign(CLp_r - CLn_r)
        red_interp = (1 + CLloc_r) * Cn_r / 2 + (1 - CLloc_r) * Cp_r / 2
        red = (red_interp + Ch_r + Cv_r) + x_masked[:, 0]
        red = red.unsqueeze(1)

        conv_b = F.conv2d(x_masked[:, 2].unsqueeze(1), self.kernels_algo2_chan, padding=(1, 1))
        conv_g_for_b = F.conv2d(green, self.kernels_algo2_green, padding=(1, 1))
        Ch_b = mask_Gb * (conv_b[:, 0] - conv_g_for_b[:, 0])
        Cv_b = mask_Gr * (conv_b[:, 1] - conv_g_for_b[:, 1])
        Cp_b = mosaic_mask[0] * (conv_b[:, 2] - conv_g_for_b[:, 2] / 4)
        Cn_b = mosaic_mask[0] * (conv_b[:, 3] - conv_g_for_b[:, 3] / 4)
        CLp_b = mosaic_mask[0] * (torch.abs(conv_b[:, 4]) + torch.abs(conv_g_for_b[:, 2]))
        CLn_b = mosaic_mask[0] * (torch.abs(conv_b[:, 5]) + torch.abs(conv_g_for_b[:, 3]))
        CLloc_b = torch.sign(CLp_b - CLn_b)
        blue_interp = (1 + CLloc_b) * Cn_b / 2 + (1 - CLloc_b) * Cp_b / 2
        blue = (blue_interp + Ch_b + Cv_b) + x_masked[:, 2]
        blue = blue.unsqueeze(1)

        output = torch.cat((red, green, blue), axis=1)
        return output


if __name__ == "__main__":
    BAYER_PATTERN = 'rggb'
    ONNX_MODEL_PATH = "hamilton_adam_demosaic.onnx"

    print("--- Step 1: Initializing PyTorch model for DYNAMIC SHAPE ONNX export ---")
    model = HamiltonAdamONNXDynamic(pattern=BAYER_PATTERN)
    model.eval()

    # Create a dummy input with a typical, non-256x256 size
    dummy_input = torch.randn(1, 4, 100, 150, requires_grad=False)

    print(f"\n--- Step 2: Exporting model to {ONNX_MODEL_PATH} ---")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_MODEL_PATH,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['bayer_input'],
        output_names=['rgb_output'],
        dynamic_axes={
            'bayer_input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'rgb_output': {0: 'batch_size', 2: 'out_height', 3: 'out_width'}
        }
    )
    print("Export complete. The new model now supports variable input sizes.")

    # --- Verification Step ---
    print("\n--- Step 3: Verifying the new model with different input sizes ---")
    ort_session = onnxruntime.InferenceSession(ONNX_MODEL_PATH)
    input_name = ort_session.get_inputs()[0].name

    # Test 1: Small size
    test_input_1 = np.random.rand(1, 4, 64, 80).astype(np.float32)
    output_1 = ort_session.run(None, {input_name: test_input_1})[0]
    print(f"Input: {test_input_1.shape} -> Output: {output_1.shape} (Correct!)")
    assert output_1.shape == (1, 3, 128, 160)

    # Test 2: HD size (like your video)
    test_input_2 = np.random.rand(1, 4, 540, 960).astype(np.float32)  # Half HD for speed
    output_2 = ort_session.run(None, {input_name: test_input_2})[0]
    print(f"Input: {test_input_2.shape} -> Output: {output_2.shape} (Correct!)")
    assert output_2.shape == (1, 3, 1080, 1920)

    print("\nVerification successful! The model is ready.")
