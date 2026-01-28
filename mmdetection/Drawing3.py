import matplotlib.pyplot as plt
import numpy as np

# ===================== Global Configuration & Data Preparation (Keep all polishing settings) =====================
# Solve Matplotlib Chinese/English rendering issues, set Times New Roman (commonly used in academic papers)
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue
plt.rcParams['font.family'] = 'serif'  # Global serif font, all text inherits this setting automatically

# Chart core configuration
algorithms = ['RTMDet', 'Faster R-CNN', 'Cascade R-CNN', 'DINO']
bar_colors = ['#FFA500', '#2E8B57']  # Academic color scheme: Warm Orange-Corrupted Test, Dark Green-Retrain Test
line_colors = ['#E65100', '#1E6F5C']  # Line colors: Darker shades of the same color system (Dark Orange-Corrupted, Dark Green-Retrain)
markers = ['s', '^']  # Line markers: Square-Corrupted Test, Triangle-Retrain Test
width = 0.32  # Fine-tune bar width for more comfortable layout
x = np.arange(len(algorithms))  # X-axis coordinates
fontsize_label = 14  # Font size of value labels on bars
line_style = '-'  # Line style: Solid line
line_width = 2  # Line width
marker_size = 7  # Marker size
marker_edge_width = 1.5  # Marker edge width
bar_edge_color = 'white'  # White edge for bars to enhance 3D effect
bar_edge_width = 1  # Bar edge width

# ---------------------- Replace with Corrupted Test vs Retrain Test data from screenshot ----------------------
# AP metrics
corrupted_ap = [0.369, 0.325, 0.362, 0.402]
retrain_ap = [0.392, 0.336, 0.380, 0.413]

# AP50 metrics
corrupted_ap50 = [0.622, 0.521, 0.586, 0.638]
retrain_ap50 = [0.665, 0.580, 0.616, 0.673]

# APs metrics
corrupted_aps = [0.273, 0.249, 0.272, 0.328]
retrain_aps = [0.282, 0.267, 0.286, 0.314]


# ===================== Universal Function: Add Value Labels on Top of Bars (Keep original settings) =====================
def add_bar_labels(ax, bars, offset=0.01):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + offset,
                f'{height:.3f}', ha='center', va='bottom', fontsize=fontsize_label,
                fontfamily='Times New Roman')


# ===================== Universal Plotting Function: Fix fontfamily error, keep other polishing effects =====================
def plot_metric_fig(metric_cor, metric_ret, title, ylim, label_offset, save_name):
    fig, ax = plt.subplots(figsize=(8, 6))

    # 1. Draw grouped bar chart (with white edge)
    bar_cor = ax.bar(x - width / 2, metric_cor, width, label='Corrupted Test', color=bar_colors[0],
                    edgecolor=bar_edge_color, linewidth=bar_edge_width)
    bar_ret = ax.bar(x + width / 2, metric_ret, width, label='Retrain Test', color=bar_colors[1],
                     edgecolor=bar_edge_color, linewidth=bar_edge_width)

    # 2. Draw lines (filled and stroked markers)
    ax.plot(x - width / 2, metric_cor, color=line_colors[0], linestyle=line_style,
            marker=markers[0], ms=marker_size, lw=line_width,
            markeredgecolor=line_colors[0], markeredgewidth=marker_edge_width,
            markerfacecolor=bar_colors[0])
    ax.plot(x + width / 2, metric_ret, color=line_colors[1], linestyle=line_style,
            marker=markers[1], ms=marker_size, lw=line_width,
            markeredgecolor=line_colors[1], markeredgewidth=marker_edge_width,
            markerfacecolor=bar_colors[1])

    # 3. Add value labels on bars
    add_bar_labels(ax, bar_cor, offset=label_offset)
    add_bar_labels(ax, bar_ret, offset=label_offset)

    # 4. Layout polishing: Core fix - Remove fontfamily from tick_params to resolve error
    ax.set_title(title, fontsize=18, fontfamily='Times New Roman', pad=15)
    ax.set_xticks(x)
    # X-axis ticks: Explicitly specify font, others inherit global settings
    ax.set_xticklabels(algorithms, fontsize=16, fontfamily='Times New Roman', rotation=0)
    ax.set_ylim(ylim)
    # Set y-axis tick step and generate tick values
    y_ticks = np.arange(ylim[0], ylim[1] + 0.05, 0.1)
    ax.set_yticks(y_ticks)
    # Y-axis ticks: Explicitly specify font (replace incorrect tick_params usage, more reliable)
    ax.set_yticklabels([f'{t:.1f}' for t in y_ticks], fontsize=16, fontfamily='Times New Roman')

    # Hide top/right spines + Light gray horizontal grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#EAEAEA', linestyle='-', linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    # Simplified legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=False,
              framealpha=1, prop={'family': 'Times New Roman', 'size': 14})

    # 5. Save as high-resolution image (300 dpi)
    plt.savefig(
        save_name,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    plt.close()


# ===================== Draw three independent charts (Call universal function, no errors) =====================
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