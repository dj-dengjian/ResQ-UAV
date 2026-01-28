import matplotlib.pyplot as plt
import numpy as np

# ===================== 全局配置 & 数据准备 (保留所有润色设置) =====================
# 解决Matplotlib中英文渲染问题，设置学术常用字体Times New Roman
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.family'] = 'serif'  # 全局衬线字体，所有文字自动继承

# 图表核心配置
algorithms = ['RTMDet', 'Faster R-CNN', 'Cascade R-CNN', 'DINO']
bar_colors = ['#FFA500', '#2E8B57']  # 学术风配色：暖橙-Corrupted Test，深绿-Retrain Test
line_colors = ['#E65100', '#1E6F5C']  # 折线颜色：同色系加深（深橙-Corrupted，深绿-Retrain）
markers = ['s', '^']  # 折线标记：方块-Corrupted，三角-Retrain
width = 0.32  # 微调柱子宽度，布局更舒展
x = np.arange(len(algorithms))  # x轴坐标
fontsize_label = 14  # 数值标注字体大小
line_style = '-'  # 折线样式：实线
line_width = 2  # 折线宽度
marker_size = 7  # 标记点大小
marker_edge_width = 1.5  # 标记点描边宽度
bar_edge_color = 'white'  # 柱子白色描边，增加立体感
bar_edge_width = 1  # 柱子描边宽度

# ---------------------- 替换为截图中的 Corrupted Test vs Retrain Test 数据 ----------------------
# AP 指标
corrupted_ap = [0.369, 0.325, 0.362, 0.402]
retrain_ap = [0.392, 0.336, 0.380, 0.413]

# AP50 指标
corrupted_ap50 = [0.622, 0.521, 0.586, 0.638]
retrain_ap50 = [0.665, 0.580, 0.616, 0.673]

# APs 指标
corrupted_aps = [0.273, 0.249, 0.272, 0.328]
retrain_aps = [0.282, 0.267, 0.286, 0.314]


# ===================== 通用函数：柱状图顶部数值标注 (保留原设置) =====================
def add_bar_labels(ax, bars, offset=0.01):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + offset,
                f'{height:.3f}', ha='center', va='bottom', fontsize=fontsize_label,
                fontfamily='Times New Roman')


# ===================== 通用绘图函数：修复fontfamily报错，其余润色效果不变 =====================
def plot_metric_fig(metric_cor, metric_ret, title, ylim, label_offset, save_name):
    fig, ax = plt.subplots(figsize=(8, 6))

    # 1. 绘制分组柱状图（加白色描边）
    bar_cor = ax.bar(x - width / 2, metric_cor, width, label='Corrupted Test', color=bar_colors[0],
                    edgecolor=bar_edge_color, linewidth=bar_edge_width)
    bar_ret = ax.bar(x + width / 2, metric_ret, width, label='Retrain Test', color=bar_colors[1],
                     edgecolor=bar_edge_color, linewidth=bar_edge_width)

    # 2. 绘制折线（标记点填充+描边）
    ax.plot(x - width / 2, metric_cor, color=line_colors[0], linestyle=line_style,
            marker=markers[0], ms=marker_size, lw=line_width,
            markeredgecolor=line_colors[0], markeredgewidth=marker_edge_width,
            markerfacecolor=bar_colors[0])
    ax.plot(x + width / 2, metric_ret, color=line_colors[1], linestyle=line_style,
            marker=markers[1], ms=marker_size, lw=line_width,
            markeredgecolor=line_colors[1], markeredgewidth=marker_edge_width,
            markerfacecolor=bar_colors[1])

    # 3. 添加数值标注
    add_bar_labels(ax, bar_cor, offset=label_offset)
    add_bar_labels(ax, bar_ret, offset=label_offset)

    # 4. 版式润色：核心修改——删除tick_params里的fontfamily，修复报错
    ax.set_title(title, fontsize=18, fontfamily='Times New Roman', pad=15)
    ax.set_xticks(x)
    # x轴刻度：显式指定字体，其余继承全局
    ax.set_xticklabels(algorithms, fontsize=16, fontfamily='Times New Roman', rotation=0)
    ax.set_ylim(ylim)
    # 设置y轴刻度步长，生成刻度值
    y_ticks = np.arange(ylim[0], ylim[1] + 0.05, 0.1)
    ax.set_yticks(y_ticks)
    # y轴刻度：显式指定字体（替代原tick_params的错误写法，更稳妥）
    ax.set_yticklabels([f'{t:.1f}' for t in y_ticks], fontsize=16, fontfamily='Times New Roman')

    # 隐藏顶/右边框 + 浅灰色水平网格
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#EAEAEA', linestyle='-', linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    # 精简图例
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=False,
              framealpha=1, prop={'family': 'Times New Roman', 'size': 14})

    # 5. 300dpi高清保存
    plt.savefig(
        save_name,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    plt.close()


# ===================== 绘制三张独立图（调用通用函数，无报错） =====================
# 1. AP (Average Precision)
plot_metric_fig(corrupted_ap, retrain_ap, 'AP (Corrupted vs Retrain Test)', (0, 0.6),
                label_offset=0.012,
                save_name='detection_algorithm_AP_corrupted_vs_retrain.png')

# 2. AP50 (AP at IoU=0.5)
plot_metric_fig(corrupted_ap50, retrain_ap50, 'AP50 (Corrupted vs Retrain Test)', (0, 0.9),
                label_offset=0.018,
                save_name='detection_algorithm_AP50_corrupted_vs_retrain.png')

# 3. APs (AP for Small Objects)
plot_metric_fig(corrupted_aps, retrain_aps, 'APs (Corrupted vs Retrain Test)', (0, 0.5),
                label_offset=0.009,
                save_name='detection_algorithm_APs_corrupted_vs_retrain.png')