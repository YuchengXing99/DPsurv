import cv2
import numpy as np
from PIL import Image, ImageOps
import argparse

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib import font_manager as fm, rcParams
from matplotlib import patches as mpatches
import seaborn as sns

from PIL import Image
# from mil_models import create_model
import numpy as np
from PIL import Image, ImageOps
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from feature_extraction.panther import PANTHERBase
from feature_extraction.tokenizer import PrototypeTokenizer
from matplotlib.ticker import MultipleLocator


def get_panther_encoder(in_dim: int, p: int, proto_path: str, **kwargs):
    """
    Build a frozen PANTHERBase encoder with a .representation() convenience method.

    Args:
        in_dim:     patch feature dimension (e.g. 1536 for UNI2)
        p:          number of GMM prototypes (e.g. 16)
        proto_path: path to prototype .pkl file
    Returns:
        encoder with .representation(feats) -> {'repr': tensor, 'qq': tensor}
    """
    model = PANTHERBase(d=in_dim, p=p, L=1, tau=1.0, out='allcat',
                        load_proto=True, proto_path=proto_path, fix_proto=True)
    model.eval()
    tokenizer = PrototypeTokenizer(proto_model_type='PANTHER', out_type='allcat', p=p)

    def representation(feats):
        with torch.inference_mode():
            h = feats.unsqueeze(0)
            flat_repr, qqs = model(h)
        return {'repr': flat_repr, 'qq': qqs}

    import torch
    model.representation = representation
    return model

def get_mixture_plot_threshold(mixtures, threshold=0.01,
                               hide_xticks=True, ymax=0.55,
                               figsize=(4, 2.5), bar_width=0.6):

    import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns
    from matplotlib import font_manager as fm

    # ---- 原始全集 label 与颜色映射（不随过滤改变） ----
    K = len(mixtures)
    labels_all = [f'c{i}' for i in range(K)]
    colors = [
        '#696969','#556b2f','#a0522d','#483d8b',
        '#008000','#008b8b','#000080','#7f007f',
        '#8fbc8f','#b03060','#ff0000','#ffa500',
        '#00ff00','#8a2be2','#00ff7f','#FFFF54',
        '#00ffff','#00bfff','#f4a460','#adff2f',
        '#da70d6','#b0c4de','#ff00ff','#1e90ff',
        '#f0e68c','#0000ff','#dc143c','#90ee90',
        '#ff1493','#7b68ee','#ffefd5','#ffb6c1'
    ]
    cmap_full = {lab: colors[i % len(colors)] for i, lab in enumerate(labels_all)}

    # ---- 数据框 & 过滤（不改映射）----
    df = pd.DataFrame({'cluster': labels_all, 'value': np.asarray(mixtures, dtype=float)})
    df = df[df['value'] > threshold]
    # 维持按原编号排序
    df = df.sort_values(key=lambda s: s.str[1:].astype(int), by='cluster')

    # ---- 画图 ----
    fig = plt.figure(figsize=figsize, dpi=300)
    prop = fm.FontProperties(fname="./Arial.ttf")
    mpl.rcParams['axes.linewidth'] = 1.0
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    ax = sns.barplot(
        data=df,
        x="cluster", y="value",
        hue="cluster", palette=cmap_full,
        legend=False, width=bar_width
    )

    # 纵轴标签（LaTeX）
    ax.set_ylabel(r'Proportion $\pi_c$', fontproperties=prop, fontsize=10, labelpad=2)

    # 横轴可选隐藏
    if hide_xticks:
        ax.set_xticks([])
        ax.set_xlabel('')
    else:
        ax.set_xlabel('Cluster', fontproperties=prop, fontsize=10)

    # y 轴范围
    if ymax is None:
        ymax = float(df['value'].max()) * 1.15 if len(df) else 0.1
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(MultipleLocator(0.2)) 
    ax.tick_params(axis='y', which='major', pad=1, length=3, width=0.8)
    fig.tight_layout(pad=0.1)
    ax.tick_params(labelsize=8)

    plt.close()
    return fig





def get_mixture_plot(mixtures):
    colors = [
        '#696969','#556b2f','#a0522d','#483d8b', 
        '#008000','#008b8b','#000080','#7f007f',
        '#8fbc8f','#b03060','#ff0000','#ffa500',
        '#00ff00','#8a2be2','#00ff7f', '#FFFF54', 
        '#00ffff','#00bfff','#f4a460','#adff2f',
        '#da70d6','#b0c4de','#ff00ff','#1e90ff',
        '#f0e68c','#0000ff','#dc143c','#90ee90',
        '#ff1493','#7b68ee','#ffefd5','#ffb6c1']

    cmap = {f'c{k}':v for k,v in enumerate(colors[:len(mixtures)])}
    mpl.rcParams['axes.spines.left'] = True
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.bottom'] = True
    fig = plt.figure(figsize=(6,3), dpi=300)

    prop = fm.FontProperties(fname="./Arial.ttf")
    mpl.rcParams['axes.linewidth'] = 1.3
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    mixtures = pd.DataFrame(mixtures, index=cmap.keys()).T
    ax = sns.barplot(mixtures, palette=cmap)
    plt.axis('on')
    plt.tick_params(axis='both', left=True, top=False, right=False, bottom=True, labelleft=True, labeltop=False, labelright=False, labelbottom=True)
    ax.set_xlabel('Cluster', fontproperties=prop, fontsize=12)
    ax.set_ylabel('Proportion / Mixture', fontproperties=prop, fontsize=12)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontproperties = prop, fontsize=12)
    ax.set_ylim([0, 0.55])
    plt.close()
    return ax.get_figure()

def hex_to_rgb_mpl_255(hex_color):
    rgb = mcolors.to_rgb(hex_color)
    return tuple([int(x*255) for x in rgb])

def get_default_cmap(n=32):
    colors = [
        '#696969','#556b2f','#a0522d','#483d8b', 
        '#008000','#008b8b','#000080','#7f007f',
        '#8fbc8f','#b03060','#ff0000','#ffa500',
        '#00ff00','#8a2be2','#00ff7f', '#FFFF54', 
        '#00ffff','#00bfff','#f4a460','#adff2f',
        '#da70d6','#b0c4de','#ff00ff','#1e90ff',
        '#f0e68c','#0000ff','#dc143c','#90ee90',
        '#ff1493','#7b68ee','#ffefd5','#ffb6c1'
    ]
    
    colors = colors[:n]
    label2color_dict = dict(zip(range(n), [hex_to_rgb_mpl_255(x) for x in colors]))
    return label2color_dict

def get_panther_encoder(in_dim, p, proto_path, config_dir='../'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='PANTHER')
    parser.add_argument('--proto_model_type', type=str, default='PANTHER')
    parser.add_argument('--model_config', type=str, default='PANTHER_default')
    parser.add_argument('--in_dim', type=int, default=in_dim)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--n_proto', type=int, default=16)
    parser.add_argument('--n_classes', type=str, default=2)
    parser.add_argument('--out_size', type=int, default=p)
    parser.add_argument('--em_iter', type=int, default=1)
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--out_type', type=str, default='allcat')
    parser.add_argument('--n_fc_layers', type=int, default=0)
    parser.add_argument('--load_proto', type=int, default=1)
    parser.add_argument('--ot_eps', type=int, default=1)
    args = parser.parse_known_args()[0]
    args.fix_proto = 1
    args.proto_path = proto_path

    model = create_embedding_model(args, config_dir=config_dir)
    model.eval()
    return model

def visualize_categorical_heatmap(
        wsi,
        coords, 
        labels, 
        label2color_dict,
        vis_level=None,
        patch_size=(256, 256),
        canvas_color=(255, 255, 255),
        alpha=0.4,
        verbose=True,
    ):

    # Scaling from 0 to desired level
    downsample = int(wsi.level_downsamples[vis_level])
    scale = [1/downsample, 1/downsample]


    if len(labels.shape) == 1:
        labels = labels.reshape(-1, 1)

    top_left = (0, 0)
    bot_right = wsi.level_dimensions[0]
    region_size = tuple((np.array(wsi.level_dimensions[0]) * scale).astype(int))
    w, h = region_size

    patch_size_orig = patch_size
    patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
    coords = np.ceil(coords * np.array(scale)).astype(int)

    if verbose:
        print('\nCreating heatmap for: ')
        print('Top Left: ', top_left, 'Bottom Right: ', bot_right)
        print('Width: {}, Height: {}'.format(w, h))
        print(f'Original Patch Size / Scaled Patch Size: {patch_size_orig} / {patch_size}')
    
    vis_level = wsi.get_best_level_for_downsample(downsample)
    img = wsi.read_region(top_left, vis_level, wsi.level_dimensions[vis_level]).convert("RGB")
    if img.size != region_size:
        img = img.resize(region_size, resample=Image.Resampling.BICUBIC)
    img = np.array(img)
    
    if verbose:
        print('vis_level: ', vis_level)
        print('downsample: ', downsample)
        print('region_size: ', region_size)
        print('total of {} patches'.format(len(coords)))
    
    for idx in tqdm(range(len(coords))):
        coord = coords[idx]
        color = label2color_dict[labels[idx][0]]
        img_block = img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]].copy()
        color_block = (np.ones((img_block.shape[0], img_block.shape[1], 3)) * color).astype(np.uint8)
        blended_block = cv2.addWeighted(color_block, alpha, img_block, 1 - alpha, 0)
        blended_block = np.array(ImageOps.expand(Image.fromarray(blended_block), border=1, fill=(50,50,50)).resize((img_block.shape[1], img_block.shape[0])))
        img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] = blended_block

    img = Image.fromarray(img)
    return img
'''
def visualize_single_class_highlight(
        wsi,
        coords,
        labels,
        label2color_dict,
        target_class,
        vis_level=None,
        patch_size=(256, 256),
        mode="tint",          # "tint" | "outline" | "dim_others"
        alpha=0.35,           # 仅对 "tint" 生效
        outline_thickness=2,  # 仅对 "outline" 生效
        dim_alpha=0.4,        # 仅对 "dim_others" 生效
        verbose=True,
    ):
    assert vis_level is not None, "Please pass a valid vis_level."

    # 缩放
    downsample = float(wsi.level_downsamples[vis_level])
    scale = 1.0 / downsample

    labels = np.asarray(labels).reshape(-1)
    coords_scaled = np.ceil(coords * scale).astype(int)
    pw, ph = np.ceil(np.array(patch_size) * scale).astype(int)
    pw, ph = int(pw), int(ph)

    # 读取底图（整张原图保留）
    base = wsi.read_region((0, 0), vis_level, wsi.level_dimensions[vis_level]).convert("RGB")
    base = np.array(base)
    H, W = base.shape[:2]

    out = base.copy()

    color = tuple(map(int, label2color_dict.get(target_class, (255, 0, 0))))

    if mode == "tint":
        # 逐块混合：只对 target_class 的 patch 做 (1-alpha)*base + alpha*color
        for idx in np.nonzero(labels == target_class)[0]:
            x, y = int(coords_scaled[idx, 0]), int(coords_scaled[idx, 1])
            x2, y2 = min(x + pw, W), min(y + ph, H)
            if x >= W or y >= H or x2 <= x or y2 <= y:
                continue

            base_block = out[y:y2, x:x2]
            color_block = np.empty_like(base_block)
            color_block[:] = color
            blended = cv2.addWeighted(base_block, 1 - alpha, color_block, alpha, 0)
            out[y:y2, x:x2] = blended

    elif mode == "outline":
        for idx in np.nonzero(labels == target_class)[0]:
            x, y = int(coords_scaled[idx, 0]), int(coords_scaled[idx, 1])
            x2, y2 = min(x + pw, W), min(y + ph, H)
            if x >= W or y >= H or x2 <= x or y2 <= y:
                continue
            cv2.rectangle(out, (x, y), (x2-1, y2-1), color, thickness=outline_thickness)

    elif mode == "dim_others":
        # 整体先变暗，然后把 target patch 恢复原亮度
        dimmed = cv2.addWeighted(base, 1.0 - dim_alpha, np.zeros_like(base), dim_alpha, 0)
        out = dimmed
        for idx in np.nonzero(labels == target_class)[0]:
            x, y = int(coords_scaled[idx, 0]), int(coords_scaled[idx, 1])
            x2, y2 = min(x + pw, W), min(y + ph, H)
            if x >= W or y >= H or x2 <= x or y2 <= y:
                continue
            out[y:y2, x:x2] = base[y:y2, x:x2]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return Image.fromarray(out)
'''

def visualize_thumbnail(
        wsi,
        vis_level=None,
        max_size=(2048, 2048),  # 控制缩略图最大尺寸
        verbose=True,
    ):

    if vis_level is None:
        # 自动选择合适的层
        vis_level = wsi.get_best_level_for_downsample(32)  

    # 读取原图在指定层的区域
    img = wsi.read_region((0, 0), vis_level, wsi.level_dimensions[vis_level]).convert("RGB")

    if verbose:
        print(f"vis_level: {vis_level}")
        print(f"原图尺寸: {wsi.level_dimensions[0]}, 缩略图尺寸: {img.size}")

    # 缩放成指定最大尺寸
    img.thumbnail(max_size, Image.Resampling.LANCZOS)

    return img


def visualize_risk_heatmap(
    wsi,
    coords,
    labels,
    label2color_dict=None,
    vis_level=0,
    coords_level=0,
    patch_size=(256, 256),
    mu_components=None,
    pi_components=None,
    cmap_name='RdBu_r',
    high_is='red',
    alpha=0.55,                    # 稍微大一点更明显
    pi_thresh=0.01,
    stretch='percentile', p_low=5, p_high=95,
    spot_sigma_frac=0.6,           # 稍大一点更连续
    ksize_mult=6,
    coords_are_centers=False,      # 关键：声明 coords 的语义
    debug=True                     # 打印覆盖率等信息
):
    # 1) 读底图（目标显示层级）
    base = wsi.read_region((0, 0), vis_level, wsi.level_dimensions[vis_level]).convert("RGB")
    base = np.asarray(base)
    H, W = base.shape[0], base.shape[1]

    # 2) 缩放 coords（源层级 -> 目标层级）
    ds_src = float(wsi.level_downsamples[coords_level])
    ds_tgt = float(wsi.level_downsamples[vis_level])
    scale  = ds_src / ds_tgt
    coords_vis = np.rint(coords * scale).astype(int)
    patch_vis  = np.maximum(1, np.rint(np.array(patch_size) * scale).astype(int))  # (w,h)
    pw, ph = int(patch_vis[0]), int(patch_vis[1])

    # 3) μ→v∈[0,1]（高μ=红）
    mu = np.asarray(mu_components, float).reshape(-1)
    if pi_components is None:
        pi = np.ones_like(mu)
    else:
        pi = np.asarray(pi_components, float).reshape(-1)

    keep = np.isfinite(mu) & np.isfinite(pi) & (pi >= pi_thresh)
    if keep.any():
        if stretch == 'percentile':
            lo, hi = np.percentile(mu[keep], [p_low, p_high])
        else:
            lo, hi = mu[keep].min(), mu[keep].max()
    else:
        lo, hi = mu.min(), mu.max()

    mu_norm = np.full_like(mu, 0.5, dtype=np.float32)
    if hi > lo:
        mu_norm = np.clip((mu - lo) / (hi - lo), 0, 1).astype(np.float32)

    v_of_label = 1.0 - mu_norm if high_is == 'red' else mu_norm

    # 4) 脉冲图（中心点 or 左上角）
    risk_imp = np.zeros((H, W), np.float32)
    w_imp    = np.zeros((H, W), np.float32)

    # 如果 coords 给的是左上角，需要加半个 patch 才到中心；如果本来就是中心，就不要再加
    cx_add = 0 if coords_are_centers else pw // 2
    cy_add = 0 if coords_are_centers else ph // 2

    labels = labels.reshape(-1)
    K = len(mu)

    n_in = 0
    for (x, y), lbl in zip(coords_vis, labels):
        if not (0 <= lbl < K):
            continue
        cx = int(x + cx_add); cy = int(y + cy_add)
        if 0 <= cx < W and 0 <= cy < H:
            v = float(v_of_label[int(lbl)])
            risk_imp[cy, cx] += v
            w_imp[cy, cx]    += 1.0
            n_in += 1

    if debug:
        print(f"[DEBUG] points inside canvas: {n_in} / {len(coords_vis)} at vis_level={vis_level}")

    # 5) 高斯扩散
    sigma = max(pw, ph) * float(spot_sigma_frac)
    ksize = int(max(3, np.ceil(ksize_mult * sigma)))
    if ksize % 2 == 0: 
        ksize += 1

    risk_smooth = cv2.GaussianBlur(risk_imp, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    w_smooth    = cv2.GaussianBlur(w_imp,   (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    wmax = float(w_smooth.max())
    if debug:
        covered = (w_smooth > 1e-8).sum()
        print(f"[DEBUG] w_smooth.max()={wmax:.6f}, covered_pixels={covered}/{H*W}")

    # 若无覆盖，直接返回原图，避免“看起来没变化”的困惑
    if wmax <= 1e-12 or covered == 0:
        if debug:
            print("[DEBUG] No coverage after scaling. Check coords_level/vis_level and coords_are_centers.")
        return Image.fromarray(base)

    # 填洞避免除零
    mask = w_smooth > 1e-8
    if not mask.all():
        fill = cv2.blur(risk_smooth, (9, 9))
        risk_smooth[~mask] = fill[~mask]
        w_smooth[~mask]    = 1.0

    risk_field = np.clip(risk_smooth / w_smooth, 0.0, 1.0)

    # 6) 上色 + 仅在覆盖区域叠加
    cmap = plt.get_cmap(cmap_name)

    risk_for_color = risk_field.copy()
    risk_for_color[mask == 0] = 0.5  # 中性
    risk_rgb = (cmap(risk_for_color)[:, :, :3] * 255).astype(np.uint8)

    # 局部透明度（羽化）
    mask_soft = (w_smooth / (wmax + 1e-8))
    mask_soft = cv2.GaussianBlur(mask_soft, (9, 9), 2.0)
    mask_soft = np.clip(mask_soft, 0.0, 1.0)
    A = (alpha * mask_soft)[..., None]

    overlay = (risk_rgb.astype(np.float32) * A + base.astype(np.float32) * (1 - A)).astype(np.uint8)
    return Image.fromarray(overlay)
'''
def visualize_risk_heatmap(
        wsi,
        coords, 
        labels, 
        label2color_dict,
        vis_level=None,
        patch_size=(256, 256),
        canvas_color=(255, 255, 255),
        alpha=0.4,
        verbose=True,
        mu_components=None,
        pi_components = None,
    ):
    # === new: 用 GRFN 的 mu 来决定每个簇的颜色：mu 小更红，mu 大更蓝 ===
    #label2color_dict = mu_to_label2color(mu_components, cmap_name='coolwarm_r')
    label2color_dict = mu_to_label2color(mu_components, pi_components)

    # Scaling from 0 to desired level
    downsample = int(wsi.level_downsamples[vis_level])
    scale = [1/downsample, 1/downsample]

    if len(labels.shape) == 1:
        labels = labels.reshape(-1, 1)

    top_left = (0, 0)
    bot_right = wsi.level_dimensions[0]
    region_size = tuple((np.array(wsi.level_dimensions[0]) * scale).astype(int))
    w, h = region_size

    patch_size_orig = patch_size
    patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
    coords = np.ceil(coords * np.array(scale)).astype(int)

    if verbose:
        print('\nCreating heatmap for: ')
        print('Top Left: ', top_left, 'Bottom Right: ', bot_right)
        print('Width: {}, Height: {}'.format(w, h))
        print(f'Original Patch Size / Scaled Patch Size: {patch_size_orig} / {patch_size}')
    
    vis_level = wsi.get_best_level_for_downsample(downsample)
    img = wsi.read_region(top_left, vis_level, wsi.level_dimensions[vis_level]).convert("RGB")
    if img.size != region_size:
        img = img.resize(region_size, resample=Image.Resampling.BICUBIC)
    img = np.array(img)
    
    if verbose:
        print('vis_level: ', vis_level)
        print('downsample: ', downsample)
        print('region_size: ', region_size)
        print('total of {} patches'.format(len(coords)))
    
    for idx in tqdm(range(len(coords))):
        coord = coords[idx]
        color = label2color_dict[int(labels[idx][0])]
        img_block = img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]].copy()
        color_block = (np.ones((img_block.shape[0], img_block.shape[1], 3)) * color).astype(np.uint8)
        blended_block = cv2.addWeighted(color_block, alpha, img_block, 1 - alpha, 0)
        #blended_block = np.array(
        #    ImageOps.expand(Image.fromarray(blended_block), border=1, fill=(50,50,50))
        #         .resize((img_block.shape[1], img_block.shape[0]))
        #)
        img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] = blended_block

    img = Image.fromarray(img)
    return img

def mu_to_label2color(mu_components,
                      pi_components,
                      cmap_name='coolwarm_r',  # 反向色带：0=红, 1=蓝 → 高mu会更蓝
                      pi_thresh=0.01,
                      other_color=(200, 200, 200),
                      tie_tol=1e-12):
    """
    返回: {label_index: (R, G, B)} (0-255整数)

    规则：
    - pi < pi_thresh 的组件统一用 other_color。
    - 其余组件按 mu 从高到低排序并映射到 [0,1]。
      使用 'coolwarm_r' 时，v=1 为蓝、v=0 为红 → 高mu更蓝，低mu更红。
    """
    mu = np.asarray(mu_components, dtype=float).reshape(-1)
    pi = np.asarray(pi_components, dtype=float).reshape(-1)
    if mu.shape != pi.shape:
        raise ValueError(f"mu 和 pi 形状不一致: {mu.shape} vs {pi.shape}")

    label2color = {}
    keep = np.isfinite(mu) & np.isfinite(pi) & (pi >= pi_thresh)
    drop_idx = np.where(~keep)[0]
    for k in drop_idx:
        label2color[int(k)] = tuple(int(c) for c in other_color)

    idx_keep = np.where(keep)[0]
    m = idx_keep.size
    if m == 0:
        return label2color

    mu_keep = mu[idx_keep]
    if m == 1:
        v = np.array([0.5], dtype=float)
    else:
        order = np.argsort(-mu_keep)   # 降序
        v_sorted = mu_keep[order]
        ranks = np.empty(m, dtype=float)
        i = 0
        while i < m:
            j = i
            while j + 1 < m and np.isclose(v_sorted[j+1], v_sorted[i], rtol=0, atol=tie_tol):
                j += 1
            avg_rank = (i + j) / 2.0
            ranks[order[i:j+1]] = avg_rank
            i = j + 1
        v = 1.0 - ranks / (m - 1)      # 高mu→1（在 coolwarm_r 中对应蓝）

    cmap = plt.get_cmap(cmap_name)
    for k, vv in zip(idx_keep, v):
        r, g, b, _ = cmap(float(vv))
        label2color[int(k)] = (int(r * 255), int(g * 255), int(b * 255))
    return label2color
'''
def mu_to_label2color(mu_components,
                      pi_components,
                      cmap_name='RdBu_r',     # 蓝↔红
                      pi_thresh=0.01,
                      other_color=(220, 220, 220),
                      tie_tol=1e-12,
                      high_is='red',          # 'red' 或 'blue'
                      stretch='percentile',   # 'none' / 'minmax' / 'percentile'
                      p_low=5, p_high=95):
    """
    若你还需要“离散上色”的版本，可保留此函数。
    这里也做了对比度拉伸（percentile），让色带“拉满”更亮。
    """
    mu = np.asarray(mu_components, float).reshape(-1)
    pi = np.asarray(pi_components, float).reshape(-1)
    if mu.shape != pi.shape:
        raise ValueError(f"mu 和 pi 形状不一致: {mu.shape} vs {pi.shape}")

    label2color = {}
    keep = np.isfinite(mu) & np.isfinite(pi) & (pi >= pi_thresh)

    mu_work = mu.copy()
    if keep.any():
        if stretch == 'minmax':
            lo, hi = mu_work[keep].min(), mu_work[keep].max()
        elif stretch == 'percentile':
            lo, hi = np.percentile(mu_work[keep], [p_low, p_high])
        else:
            lo, hi = mu_work[keep].min(), mu_work[keep].max()
        if hi > lo:
            mu_work = np.clip((mu_work - lo) / (hi - lo), 0, 1)
        else:
            mu_work[:] = 0.5

    drop_idx = np.where(~keep)[0]
    for k in drop_idx:
        label2color[int(k)] = tuple(map(int, other_color))

    idx_keep = np.where(keep)[0]
    m = idx_keep.size
    if m == 0:
        return label2color

    mu_keep = mu_work[idx_keep]
    if m == 1:
        v = np.array([0.5], float)
    else:
        order = np.argsort(-mu_keep)
        v_sorted = mu_keep[order]
        ranks = np.empty(m, float)
        i = 0
        while i < m:
            j = i
            while j + 1 < m and np.isclose(v_sorted[j+1], v_sorted[i], rtol=0, atol=tie_tol):
                j += 1
            avg_rank = (i + j) / 2.0
            ranks[order[i:j+1]] = avg_rank
            i = j + 1
        # RdBu_r 中 v 越小越红 → 高风险=红
        v = ranks / (m - 1) if high_is == 'red' else 1.0 - ranks / (m - 1)

    cmap = plt.get_cmap(cmap_name)
    for k, vv in zip(idx_keep, v):
        r, g, b, _ = cmap(float(vv))
        label2color[int(k)] = (int(r*255), int(g*255), int(b*255))
    return label2color

import numpy as np
import cv2
from PIL import Image

def visualize_single_class_highlight(
        wsi,
        coords,
        labels,
        label2color_dict,
        target_class,
        vis_level=None,
        patch_size=(256, 256),
        mode="tint",            # "tint" | "outline" | "dim_others"
        alpha=0.35,             # 仅对 "tint" 生效
        outline_thickness=2,    # 仅对 "outline" 生效
        dim_alpha=0.4,          # 仅对 "dim_others" 生效
        feather_sigma=2.0,      # <<< 新增：边缘平滑(高斯羽化)的σ，0=关闭
        outline_color=None,     # None=用 label2color_dict[target_class]
        verbose=True,
    ):
    assert vis_level is not None, "Please pass a valid vis_level."

    # === 缩放 ===
    downsample = float(wsi.level_downsamples[vis_level])
    scale = 1.0 / downsample

    labels = np.asarray(labels).reshape(-1)
    coords_scaled = np.ceil(coords * scale).astype(int)
    pw, ph = np.ceil(np.array(patch_size) * scale).astype(int)
    pw, ph = int(max(pw,1)), int(max(ph,1))

    # === 读取底图（整张原图保留）===
    base = wsi.read_region((0, 0), vis_level, wsi.level_dimensions[vis_level]).convert("RGB")
    base = np.array(base)
    H, W = base.shape[:2]
    out = base.copy()

    # === 颜色 ===
    color = tuple(map(int, label2color_dict.get(target_class, (255, 0, 0))))
    if outline_color is None:
        outline_color = color

    # === 构建 target 的整图 mask (0/255) ===
    mask = np.zeros((H, W), dtype=np.uint8)
    idxs = np.nonzero(labels == target_class)[0]
    for idx in idxs:
        x, y = int(coords_scaled[idx, 0]), int(coords_scaled[idx, 1])
        x2, y2 = min(x + pw, W), min(y + ph, H)
        if x >= W or y >= H or x2 <= x or y2 <= y:
            continue
        mask[y:y2, x:x2] = 255

    if verbose:
        nz = int(mask.sum() // 255)
        print(f"[feather] target patches: {len(idxs)}, mask pixels: {nz}")

    # === 软掩码：高斯羽化 (0..1) ===
    if feather_sigma and feather_sigma > 0:
        mask_soft = cv2.GaussianBlur(mask, (0, 0), feather_sigma).astype(np.float32) / 255.0
    else:
        mask_soft = (mask.astype(np.float32) / 255.0)

    mask3 = np.dstack([mask_soft]*3)  # (H,W,3), 0..1

    if mode == "tint":
        # 只在 target 区域进行逐像素混合：out = base*(1 - a*m) + color*(a*m)
        color_img = np.empty_like(base); color_img[:] = color
        a = float(alpha)
        w = (a * mask3).astype(np.float32)
        out = (base.astype(np.float32) * (1.0 - w) + color_img.astype(np.float32) * w).clip(0,255).astype(np.uint8)

    elif mode == "outline":
        # 用形态学梯度从二值mask取边缘，再画线（边缘本身已被羽化柔和）
        edges = cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_GRADIENT,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
        ys, xs = np.where(edges > 0)
        out[ys, xs] = outline_color
        # 也可用 cv2.rectangle 逐patch画框（硬边），保留你的老逻辑：
        # for idx in idxs:
        #     x, y = int(coords_scaled[idx, 0]), int(coords_scaled[idx, 1])
        #     x2, y2 = min(x + pw, W), min(y + ph, H)
        #     if x >= W or y >= H or x2 <= x or y2 <= y: continue
        #     cv2.rectangle(out, (x, y), (x2-1, y2-1), outline_color, thickness=outline_thickness)

    elif mode == "dim_others":
        # 先得到暗化版本，然后用软掩码把目标区域平滑替换回原图
        dimmed = cv2.addWeighted(base, 1.0 - dim_alpha, np.zeros_like(base), dim_alpha, 0)
        out = (dimmed.astype(np.float32) * (1.0 - mask3) + base.astype(np.float32) * mask3).clip(0,255).astype(np.uint8)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return Image.fromarray(out)
