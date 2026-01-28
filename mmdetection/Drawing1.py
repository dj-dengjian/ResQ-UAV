import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba


def load_json_logs(json_log_path):
    """Load MMDetection JSON log file (supports single-line/entire JSON format)"""
    # Add: File existence check
    if not os.path.exists(json_log_path):
        raise FileNotFoundError(f"Log file not found: {json_log_path}\nPlease check the file path!")

    log_data = []
    with open(json_log_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line.strip():
                try:
                    log_item = json.loads(line.strip())
                    log_data.append(log_item)
                except json.JSONDecodeError:
                    f.seek(0)
                    log_data = json.load(f)
                    break
    return log_data


def print_all_metrics(json_log_path):
    """Print all available metric names in the log (for debugging)"""
    try:
        log_dicts = load_json_logs(json_log_path)
    except FileNotFoundError as e:
        print(f" Error loading {json_log_path}: {e}")
        return  # Skip non-existent files without interrupting the program

    all_metrics = set()
    for idx, log_item in enumerate(log_dicts[:20]):  # Only print first 20 entries for clarity
        all_metrics.update(log_item.keys())
        # Print validation entries (contain coco/ metrics)
        if any(key.startswith('coco/') for key in log_item.keys()):
            print(f"Validation log entry {idx}: {log_item}")
    print("=" * 50)
    print("All available metrics in log:")
    for metric in sorted(all_metrics):
        print(f"- {metric}")
    print("=" * 50)


def calculate_moving_stats(seq, window_size=5):
    """Calculate moving average + standard deviation (for loss/precision metrics)"""
    # Filter NaN values first
    seq = [x for x in seq if not np.isnan(x) and x is not None]
    if len(seq) < 2:
        return seq, [0.01] * len(seq)

    df = pd.DataFrame({'value': seq})
    ma_seq = df['value'].rolling(window=window_size, min_periods=1, center=True).mean().values
    ma_std = df['value'].rolling(window=window_size, min_periods=1, center=True).std().values
    ma_std = np.nan_to_num(ma_std, nan=0.01)
    ma_std[ma_std == 0] = np.mean(ma_std[ma_std > 0]) if np.any(ma_std > 0) else 0.01
    return ma_seq, ma_std


# ===================== Core Configuration: All models use RTMDet's red/blue color scheme uniformly =====================
# Basic metric configuration (label only)
BASE_METRIC_CONFIG = {
    'coco/bbox_mAP': {'label': 'mAP'},
    'coco/bbox_mAP_50': {'label': 'mAP50'},
    'coco/bbox_mAP_s': {'label': 'mAPs'}
}

# Add: Different marker shapes for different metrics
METRIC_MARKERS = {
    'coco/bbox_mAP': 'o',      # mAP - Keep circle
    'coco/bbox_mAP_50': 's',   # mAP50 - Use square
    'coco/bbox_mAP_s': '^'     # mAPs - Use upward triangle
}

# mAP color configuration for all models (Unified red/blue: clean=red series, corrupted=blue series)
MODEL_METRIC_COLORS = {
    # All clean models - Use RTMDet clean's red color scheme uniformly
    'RTMDet': {
        'coco/bbox_mAP': '#FF0000',  # Pure red - mAP
        'coco/bbox_mAP_50': '#FF7F00',  # Orange red - mAP50
        'coco/bbox_mAP_s': '#800080'  # Dark purple - mAPs
    },
    'Faster R-CNN': {
        'coco/bbox_mAP': '#FF0000',  # Pure red - mAP
        'coco/bbox_mAP_50': '#FF7F00',  # Orange red - mAP50
        'coco/bbox_mAP_s': '#800080'  # Dark purple - mAPs
    },
    'Cascade R-CNN': {
        'coco/bbox_mAP': '#FF0000',  # Pure red - mAP
        'coco/bbox_mAP_50': '#FF7F00',  # Orange red - mAP50
        'coco/bbox_mAP_s': '#800080'  # Dark purple - mAPs
    },
    'DINO': {
        'coco/bbox_mAP': '#FF0000',  # Pure red - mAP
        'coco/bbox_mAP_50': '#FF7F00',  # Orange red - mAP50
        'coco/bbox_mAP_s': '#800080'  # Dark purple - mAPs
    },
    # All corrupted models - Use RTMDet corrupted's blue color scheme uniformly
    'RTMDet_corrupted': {
        'coco/bbox_mAP': '#0000FF',  # Pure blue - mAP
        'coco/bbox_mAP_50': '#007FFF',  # Light blue - mAP50
        'coco/bbox_mAP_s': '#00FFFF'  # Cyan blue - mAPs
    },
    'Faster R-CNN_corrupted': {
        'coco/bbox_mAP': '#0000FF',  # Pure blue - mAP
        'coco/bbox_mAP_50': '#007FFF',  # Light blue - mAP50
        'coco/bbox_mAP_s': '#00FFFF'  # Cyan blue - mAPs
    },
    'Cascade R-CNN_corrupted': {
        'coco/bbox_mAP': '#0000FF',  # Pure blue - mAP
        'coco/bbox_mAP_50': '#007FFF',  # Light blue - mAP50
        'coco/bbox_mAP_s': '#00FFFF'  # Cyan blue - mAPs
    },
    'DINO_corrupted': {
        'coco/bbox_mAP': '#0000FF',  # Pure blue - mAP
        'coco/bbox_mAP_50': '#007FFF',  # Light blue - mAP50
        'coco/bbox_mAP_s': '#00FFFF'  # Cyan blue - mAPs
    }
}

# Color scheme for loss curves/violin plots (Unified: all clean=pure red, all corrupted=pure blue)
DATA_SOURCE_COLORS = {
    # All pre-corruption (clean) models - Unified pure red
    'RTMDet': '#FF0000',
    'Faster R-CNN': '#FF0000',
    'Cascade R-CNN': '#FF0000',
    'DINO': '#FF0000',
    # All post-corruption (corrupted) models - Unified pure blue
    'RTMDet_corrupted': '#0000FF',
    'Faster R-CNN_corrupted': '#0000FF',
    'Cascade R-CNN_corrupted': '#0000FF',
    'DINO_corrupted': '#0000FF'
}

# Line style configuration (All corrupted models use solid line)
CORRUPTED_LINE_STYLES = {
    'RTMDet_corrupted': '-',
    'Faster R-CNN_corrupted': '-',
    'Cascade R-CNN_corrupted': '-',
    'DINO_corrupted': '-'
}


# -------------------------- Loss Related Plotting --------------------------
def plot_combined_loss_curve(json_log_paths, labels, out_path, title_suffix):
    """Plot combined loss curve for multiple models (supports clean/corrupted grouping)
    Add: X-axis epochs/iters are truncated to the shortest model data length for alignment
    Add: Y-axis intelligent auto-scaling based on all loss data with 5% margin
    """
    plt.figure(figsize=(14, 7))  # Enlarge canvas for multiple models

    # Store X-axis and Loss data of all models for subsequent truncation
    all_model_data = []
    x_axis_type = None

    # Step 1: Extract raw data of all models
    for idx, (json_log_path, label) in enumerate(zip(json_log_paths, labels)):
        if not os.path.exists(json_log_path):
            print(f" Skip {label}: File not found - {json_log_path}")
            continue

        try:
            log_dicts = load_json_logs(json_log_path)
        except Exception as e:
            print(f" Skip {label}: Failed to load log - {e}")
            continue

        loss_seq = []
        x_seq = []

        # Extract loss and x-axis (epoch/iter)
        for log_item in log_dicts:
            if 'loss' in log_item and not np.isnan(log_item['loss']):
                if 'epoch' in log_item:
                    loss_seq.append(log_item['loss'])
                    x_seq.append(log_item['epoch'])
                    x_axis_type = 'Epoch'
                elif 'iter' in log_item:
                    loss_seq.append(log_item['loss'])
                    x_seq.append(log_item['iter'])
                    x_axis_type = 'Iteration'

        if not loss_seq:
            print(f" Skip {label}: No loss data found in log!")
            continue

        all_model_data.append({
            'label': label,
            'x_seq': x_seq,
            'loss_seq': loss_seq
        })

    if not all_model_data:
        print(f" No valid loss data for {title_suffix} models! Skip plotting loss curve.")
        return

    # Step 2: Find the shortest X-axis length and truncate all data uniformly
    min_length = min([len(data['x_seq']) for data in all_model_data])
    print(f"  Truncating all loss data to {min_length} points (shortest length)")

    # Pre-calculate all loss data for Y-axis intelligent scaling
    all_loss_values = []
    truncated_model_data = []
    for data in all_model_data:
        label = data['label']
        x_seq = data['x_seq'][:min_length]
        loss_seq = data['loss_seq'][:min_length]

        # Calculate moving average and standard deviation
        ma_loss, ma_std = calculate_moving_stats(loss_seq, window_size=10)

        # Collect all loss values (including moving average and shadow range)
        all_loss_values.extend(ma_loss)
        all_loss_values.extend(ma_loss - ma_std * 5)
        all_loss_values.extend(ma_loss + ma_std * 5)

        # Save truncated data to avoid repeated calculation
        truncated_model_data.append({
            'label': label,
            'x_seq': x_seq,
            'ma_loss': ma_loss,
            'ma_std': ma_std
        })

    # Step 3: Plot truncated data
    for data in truncated_model_data:
        label = data['label']
        x_seq = data['x_seq']
        ma_loss = data['ma_loss']
        ma_std = data['ma_std']

        # Get color and line style for current model
        loss_color = DATA_SOURCE_COLORS.get(label, f'C{idx}')
        linestyle = CORRUPTED_LINE_STYLES.get(label, '-')  # All corrupted models use solid line
        alpha = 1.0  # Unified transparency

        # Plot curve with shadow
        line = plt.plot(
            x_seq, ma_loss,
            label=f'{label}',
            color=loss_color,
            linestyle=linestyle,
            alpha=alpha,
            linewidth=1.8
        )[0]
        # Fill with the actual color of the line (avoid errors from manual color specification)
        plt.fill_between(
            x_seq,
            ma_loss - (ma_std * 5),
            ma_loss + (ma_std * 5),
            color=to_rgba(line.get_color(), alpha=0.15),
            linewidth=0
        )

    # Core Modification: Y-axis intelligent auto-scaling (5% margin reserved, ensure minimum value is 0)
    if all_loss_values:
        min_loss = min(all_loss_values)
        max_loss = max(all_loss_values)
        # Calculate 5% margin, use 0.1 as minimum margin if data range is too small
        loss_range = max_loss - min_loss
        margin = loss_range * 0.05 if loss_range > 0 else 0.1

        # Set Y-axis range, ensure the lower limit is not less than 0 (loss cannot be negative)
        y_min = max(min_loss - margin, 0)
        y_max = max_loss + margin
        plt.ylim(y_min, y_max)

    # Style optimization
    plt.title(f'Model Training Loss Curve ({title_suffix})', fontsize=24)
    plt.xlabel(x_axis_type if x_axis_type else 'Epoch/Iter', fontsize=20)
    plt.ylabel('Loss value', fontsize=20)
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.legend(fontsize=22, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Combined loss curve ({title_suffix}) saved to: {out_path}")


def plot_combined_loss_violin(json_log_paths, labels, out_path, title_suffix):
    """Plot combined loss violin/box plot for multiple models (supports clean/corrupted grouping)"""
    plt.figure(figsize=(10, 7))
    loss_data = []
    plot_labels = []
    plot_colors = []

    # Collect loss data for each source
    for idx, (json_log_path, label) in enumerate(zip(json_log_paths, labels)):
        # Add: Skip non-existent files
        if not os.path.exists(json_log_path):
            print(f"  Skip {label}: File not found - {json_log_path}")
            continue

        try:
            log_dicts = load_json_logs(json_log_path)
        except Exception as e:
            print(f"  Skip {label}: Failed to load log - {e}")
            continue

        loss_seq = [item['loss'] for item in log_dicts if 'loss' in item and not np.isnan(item['loss'])]

        if not loss_seq:
            print(f"  Skip {label}: No loss data found in log!")
            continue

        loss_data.append(loss_seq)
        # Fix: Process label first, then splice line breaks (avoid backslashes in f-string)
        processed_label = label.replace('_corrupted', 'corrupted')
        plot_labels.append(processed_label.replace('corrupted', '\n corrupted'))
        # Force use of specified colors
        plot_colors.append(DATA_SOURCE_COLORS.get(label, f'C{idx}'))

    if not loss_data:
        print(f" No valid loss data for {title_suffix} models! Skip plotting violin plot.")
        return

    # Plot violin plot
    parts = plt.violinplot(
        loss_data, positions=range(len(loss_data)), widths=0.8,
        showmeans=False, showextrema=False, showmedians=False
    )

    # Customize violin bodies
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(plot_colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)  # Unified transparency

    # Overlay box plot
    box_plot = plt.boxplot(
        loss_data, positions=range(len(loss_data)), widths=0.2,
        patch_artist=True,
        boxprops={'edgecolor': 'black', 'linewidth': 1.5},
        medianprops={'color': 'black'},
        whiskerprops={'color': 'black'},
        capprops={'color': 'black'}
    )

    # Fill box plot with colors (Precise matching of box and color)
    for i, box in enumerate(box_plot['boxes']):
        box.set_facecolor(plot_colors[i])

    # Optional: Y-axis intelligent auto-scaling for violin plot (same logic as curve)
    all_violin_loss = [loss for sublist in loss_data for loss in sublist]
    if all_violin_loss:
        min_loss = min(all_violin_loss)
        max_loss = max(all_violin_loss)
        loss_range = max_loss - min_loss
        margin = loss_range * 0.05 if loss_range > 0 else 0.1
        y_min = max(min_loss - margin, 0)
        y_max = max_loss + margin
        plt.ylim(y_min, y_max)

    # Style optimization
    plt.title(f'Model Loss Distribution ({title_suffix})', fontsize=24)
    plt.ylabel('Loss Value', fontsize=20)
    plt.tick_params(axis='y', labelsize=18)
    plt.xticks(range(len(plot_labels)), plot_labels, fontsize=18)
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Combined loss violin plot ({title_suffix}) saved to: {out_path}")


# -------------------------- Precision Metric Related Plotting --------------------------
def plot_combined_mAP_curve(json_log_paths, labels, out_path, title_suffix):
    """Plot combined mAP curve for multiple models (supports clean/corrupted grouping)
    Add: X-axis epochs/iters are truncated to the shortest model data length for alignment
    Add: All models force solid lines, different metrics use different colors
    """
    plt.figure(figsize=(14, 7))  # Enlarge canvas for multiple models

    # Store X-axis and mAP data of all models for subsequent truncation
    all_model_metric_data = []
    x_axis_type = None

    # Step 1: Extract raw mAP data of all models
    for src_idx, (json_log_path, label) in enumerate(zip(json_log_paths, labels)):
        if not os.path.exists(json_log_path):
            print(f"  Skip {label}: File not found - {json_log_path}")
            continue

        try:
            log_dicts = load_json_logs(json_log_path)
        except Exception as e:
            print(f"  Skip {label}: Failed to load log - {e}")
            continue

        # Extract validation entries
        val_entries = [item for item in log_dicts if any(key.startswith('coco/') for key in item.keys())]
        if len(val_entries) == 0:
            print(f"  Skip {label}: No validation entries (coco/ metrics) found!")
            continue

        # Extract raw data for each metric
        for metric_key, config in BASE_METRIC_CONFIG.items():
            values = []
            xs = []
            for entry in val_entries:
                if metric_key in entry:
                    val = entry[metric_key]
                    if val is not None and not np.isnan(val) and val >= 0:
                        values.append(val)
                        if 'epoch' in entry:
                            xs.append(entry['epoch'])
                            x_axis_type = 'Epoch'
                        elif 'iter' in entry:
                            xs.append(entry['iter'])
                            x_axis_type = 'Iteration'
                        else:
                            xs.append(len(xs) + 1)

            if not values:
                print(f"  Skip {label} - {config['label']}: No valid values found!")
                continue

            all_model_metric_data.append({
                'label': label,
                'metric_key': metric_key,
                'metric_label': config['label'],
                'xs': xs,
                'values': values
            })

    if not all_model_metric_data:
        print(f" No valid mAP data for {title_suffix} models! Skip plotting mAP curve.")
        return

    # Step 2: Find the shortest X-axis length and truncate all data uniformly
    min_length = min([len(data['xs']) for data in all_model_metric_data])
    print(f"  Truncating all mAP data to {min_length} points (shortest length)")

    # Step 3: Plot truncated data (force solid lines, use MODEL_METRIC_COLORS for color scheme)
    for data in all_model_metric_data:
        label = data['label']
        metric_key = data['metric_key']
        metric_label = data['metric_label']

        # Truncate to the shortest length
        xs = data['xs'][:min_length]
        values = data['values'][:min_length]

        # Calculate moving average
        if len(values) >= 5:
            ma_vals, ma_std = calculate_moving_stats(values, window_size=5)
        else:
            ma_vals, ma_std = values, [0.01] * len(values)

        # Core Modification: All models force solid lines, use unified red/blue color scheme
        if label in MODEL_METRIC_COLORS:
            base_color = MODEL_METRIC_COLORS[label][metric_key]
            linestyle = '-'  # Force solid line
        else:
            base_color = f'C{src_idx}'
            linestyle = '-'

        alpha = 1.0
        linewidth = 3.0
        marker_size = 12
        marker_interval = 2

        # Plot curve with unique label (model + metric)
        plot_label = f"{label} - {metric_label}"
        line = plt.plot(
            xs, ma_vals,
            label=plot_label,
            color=base_color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            marker=METRIC_MARKERS.get(metric_key, 'o'),
            markersize=marker_size,
            markevery=marker_interval,
            markerfacecolor=base_color,
            markeredgecolor='black',
            markeredgewidth=1.2
        )[0]

        # Plot shadow (use the actual color of the line to avoid manual specification errors)
        plt.fill_between(
            xs,
            np.array(ma_vals) - np.array(ma_std) * 5.5,
            np.array(ma_vals) + np.array(ma_std) * 5.5,
            color=to_rgba(line.get_color(), alpha=0.1),
            linewidth=0
        )

    # Style optimization
    plt.title(f'Model mAP Curve ({title_suffix})', fontsize=24)
    plt.xlabel(x_axis_type if x_axis_type else 'Epoch', fontsize=20)
    plt.ylabel('mAP Value', fontsize=20)
    plt.ylim(0, 1.0)  # mAP range is fixed 0-1, no modification needed
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.legend(fontsize=16, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.17))

    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Combined mAP curve ({title_suffix}) saved to: {out_path}")


def plot_combined_mAP_violin(json_log_paths, labels, out_path, title_suffix):
    """Plot combined mAP violin/box plot for multiple models (supports clean/corrupted grouping)"""
    plt.figure(figsize=(12, 7))
    all_violin_data = []
    all_plot_labels = []
    all_colors = []

    # Collect data for each source + metric combination
    for src_idx, (json_log_path, label) in enumerate(zip(json_log_paths, labels)):
        # Add: Skip non-existent files
        if not os.path.exists(json_log_path):
            print(f"  Skip {label}: File not found - {json_log_path}")
            continue

        try:
            log_dicts = load_json_logs(json_log_path)
        except Exception as e:
            print(f"  Skip {label}: Failed to load log - {e}")
            continue

        val_entries = [item for item in log_dicts if any(key.startswith('coco/') for key in item.keys())]
        if len(val_entries) == 0:
            print(f"  Skip {label}: No validation entries (coco/ metrics) found!")
            continue

        # Extract each metric's data
        for metric_key, config in BASE_METRIC_CONFIG.items():
            values = []
            for entry in val_entries:
                if metric_key in entry:
                    val = entry[metric_key]
                    if val is not None and not np.isnan(val) and val >= 0:
                        values.append(val)

            if values:
                all_violin_data.append(values)
                # Fix: Process label first (move backslashes out of f-string)
                processed_label = label.replace('_corrupted', 'corrupted')
                processed_label = processed_label.replace('corrupted', '\n corrupted')
                # Splice the final label
                display_label = f"{processed_label}\n{config['label']}"
                all_plot_labels.append(display_label)
                # Use unified red/blue color scheme
                if label in MODEL_METRIC_COLORS and metric_key in MODEL_METRIC_COLORS[label]:
                    all_colors.append(MODEL_METRIC_COLORS[label][metric_key])
                else:
                    all_colors.append(DATA_SOURCE_COLORS.get(label, f'C{src_idx}'))

    if not all_violin_data:
        print(f" No valid mAP data for {title_suffix} models! Skip plotting violin plot.")
        return

    # Plot violin plot
    parts = plt.violinplot(
        all_violin_data, positions=range(len(all_violin_data)), widths=0.8,
        showmeans=True, showextrema=True, showmedians=True
    )

    # Customize violin bodies
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(all_colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)

    # Customize statistical lines
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
        if partname in parts:
            plt.setp(parts[partname], color='black', linewidth=1.5)

    # Overlay box plot
    box_plot = plt.boxplot(
        all_violin_data, positions=range(len(all_violin_data)), widths=0.2,
        patch_artist=True,
        boxprops={'edgecolor': 'black', 'linewidth': 1.5},
        medianprops={'color': 'white', 'linewidth': 2},
        whiskerprops={'color': 'black', 'linewidth': 1.5},
        capprops={'color': 'black', 'linewidth': 1.5}
    )

    # Fill box plot with colors
    for i, box in enumerate(box_plot['boxes']):
        box.set_facecolor(all_colors[i])
        box.set_alpha(0.9)

    # Style optimization
    plt.title(f'Model mAP Distribution ({title_suffix})', fontsize=24)
    plt.ylabel('mAP Value', fontsize=20)
    plt.ylim(0, 1.0)  # mAP range is fixed 0-1, no modification needed
    plt.tick_params(axis='y', labelsize=18)
    plt.xticks(range(len(all_plot_labels)), all_plot_labels, fontsize=16)
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Combined mAP violin plot ({title_suffix}) saved to: {out_path}")


def plot_single_model_pair(model_name, clean_log_path, corrupted_log_path, output_dir):
    """Plot all charts for a single model pair (clean + corrupted)"""
    # Define model paths and labels
    log_paths = [clean_log_path, corrupted_log_path]
    labels = [model_name, f"{model_name}_corrupted"]
    title_suffix = f"{model_name}"

    # Define output paths
    loss_curve_path = os.path.join(output_dir, f"{model_name}_combined_loss_curve.png")
    loss_violin_path = os.path.join(output_dir, f"{model_name}_combined_loss_violin.png")
    mAP_curve_path = os.path.join(output_dir, f"{model_name}_combined_mAP_curve.png")
    mAP_violin_path = os.path.join(output_dir, f"{model_name}_combined_mAP_violin.png")

    # Print model metrics (for debugging)
    print("=" * 80 + f"\n[{model_name} Models Metrics]\n" + "=" * 80)
    for path, label in zip(log_paths, labels):
        print(f"\n--- Metrics for {label} ---")
        print_all_metrics(path)

    # Plot charts
    print("\n" + "=" * 80 + f"\nPlotting Combined {model_name} Models\n" + "=" * 80)
    try:
        plot_combined_loss_curve(log_paths, labels, loss_curve_path, title_suffix)
        plot_combined_loss_violin(log_paths, labels, loss_violin_path, title_suffix)
        plot_combined_mAP_curve(log_paths, labels, mAP_curve_path, title_suffix)
        plot_combined_mAP_violin(log_paths, labels, mAP_violin_path, title_suffix)
    except Exception as e:
        print(f"\n Failed to plot combined {model_name} models: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Output directory
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    # ===================== Model Configuration: Paths for all model pairs =====================
    model_pairs = {
        # RTMDet
        'RTMDet': {
            'clean': r"./tools/work_dirs/rtmdet/20260108_024513/vis_data/20260108_024513.json",
            'corrupted': r"./tools/work_dirs/rtmdet_corrupted/20260115_225330/vis_data/20260115_225330.json"
        },
        # Faster R-CNN
        'Faster R-CNN': {
            'clean': r"./tools/work_dirs/faster_rcnn/20260121_120601/vis_data/20260121_120601.json",
            'corrupted': r"./tools/work_dirs/faster_rcnn_corrupted/20260121_183510/vis_data/20260121_183510.json"
        },
        # Cascade R-CNN
        'Cascade R-CNN': {
            'clean': r"./tools/work_dirs/cascade_rcnn/20260125_145601/vis_data/20260125_145601.json",
            'corrupted': r"./tools/work_dirs/cascade_rcnn_corrupted/20260125_231752/vis_data/20260125_231752.json"
        },
        # DINO
        'DINO': {
            'clean': r"./tools/work_dirs/dino/20260122_111748/vis_data/20260122_111748.json",
            'corrupted': r"./tools/work_dirs/dino_corrupted/20260123_234352/vis_data/20260123_234352.json"
        }
    }

    # Plot charts for each model pair one by one
    for model_name, paths in model_pairs.items():
        plot_single_model_pair(model_name, paths['clean'], paths['corrupted'], output_dir)

    print("\n" + "=" * 80 + "\nAll plotting processes completed! Check ./outputs/ for results." + "\n" + "=" * 80)