import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

def calculate_psnr(img1, img2):
    return cv2.PSNR(img1, img2)

def display_comparison(clip_frame_info, patch_coords_list, patch_size=(256, 256), n_rows=4, save_name = 'comparison.png'):
    """
    展示视频降噪结果对比

    参数:
        clip_frame_info: 列表，每个元素为 (clip_idx, frame_idx) 元组
        patch_coords_list: [(x1,y1), (x2,y2)...] 每张图的截取区域左上角坐标列表
        patch_size: (width,height) 统一的截取块大小
        n_rows: 显示行数
    """
    n_images = len(clip_frame_info)
    # 调整 figsize 参数来设置 figure 窗口的尺寸，增加高度以容纳标题
    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 2 * n_rows + 0.7))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    titles = ["GT", "Noisy", "MoCE-JDD", "FastDVDNet-Mini"]
    # 将标题设置在每列的最上方子图上
    for col in range(4):
        axes[0, col].set_title(titles[col], fontsize=12)

    for row in range(n_rows):
        if row >= n_images:
            break
        clip_idx, frame_idx = clip_frame_info[row]
        # 将 clip_idx 填充为 3 位，frame_idx 填充为 8 位
        padded_clip_idx = f"{clip_idx:03d}"
        padded_frame_idx = f"{frame_idx:08d}"
        frame_name = f"{padded_frame_idx}.png"
        x, y = patch_coords_list[row]
        w, h = patch_size

        img_paths = [
            os.path.join(r"H:\datasets\JDD\REDS120\val\val_orig", padded_clip_idx, frame_name),
            os.path.join(r"../results/MoCE_JDD/input", padded_clip_idx, frame_name),
            os.path.join(r"../results/MoCE_JDD/exp16_netdepcv_L1_fft", padded_clip_idx, frame_name),
            os.path.join(r"../results/MoCE_JDD/exp_fastDVDnetMini", padded_clip_idx, frame_name)
        ]

        gt_img = None
        for col in range(4):
            img_path = img_paths[col]

            if not os.path.exists(img_path):
                print(f"图片不存在: {img_path}")
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"图片读取失败: {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 裁剪图片
            img = img[y:y + h, x:x + w]

            if col == 0:  # GT 图片
                gt_img = img.copy()

            ax = axes[row, col]
            ax.imshow(img)
            ax.axis('off')

            if col > 0 and gt_img is not None:
                psnr = calculate_psnr(gt_img, img)
                psnr_text = f"{psnr:.2f}dB"
                font_size = 8
                # 根据字体大小调整 pad 值
                pad_value = font_size * 0.02
                ax.text(0.02, 0.02, psnr_text, transform=ax.transAxes, color='black',
                        fontsize=font_size, verticalalignment='bottom', horizontalalignment='left',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=pad_value, boxstyle=f'round,pad={pad_value}'))

    plt.tight_layout(h_pad=0.1, w_pad=0.3)
    # 设置 top 参数为 0.95
    plt.subplots_adjust(top=0.95)

    save_path =  save_name
    # 保存图片并设置 DPI 为 500
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    print(f"图片已保存至 {save_path}")

    plt.show()

if __name__ == "__main__":

    # group1
    clip_frame_info = [
        (0, 12), (27, 24), (3, 32), (4, 40)
    ]
    patch_coords_list = [
        (50, 200), (150, 300), (200, 200), (100, 100)
    ]
    patch_size = (300, 200)

    display_comparison(clip_frame_info, patch_coords_list, patch_size, n_rows=4, save_name = 'comparison1.png')

    # group2
    clip_frame_info = [
        (5, 12), (17, 24), (8, 32), (14, 40)
    ]
    patch_coords_list = [
        (50, 200), (150, 300), (200, 200), (100, 100)
    ]
    patch_size = (300, 200)

    display_comparison(clip_frame_info, patch_coords_list, patch_size, n_rows=4, save_name = 'comparison2.png')