import json
import os

def update_image_to_signal():
    path = "/home/me/mini-proj/supervisor_notebooks/learning_image_to_signal.ipynb"
    if not os.path.exists(path): return
    with open(path, 'r') as f: nb = json.load(f)
    nb['cells'][0]['source'] = [
        "# 🩺 Path 2: Image-to-Signal (ECG Digitization & Reconstruction)\n",
        "\n",
        "**Research Context**: Legacy clinical systems often rely on paper-based records. This path focuses on 'Neural Digitization'—reconstructing high-fidelity 1D electrical signals from 2D visual representations.\n",
        "\n",
        "### 🌟 Supervisor's Guide to this Notebook:\n",
        "1.  **Transformation Pipeline**: See how we simulate real-world variability (noise, tilt, contrast) to make the model robust.\n",
        "2.  **Neural Vision**: A step-by-step breakdown of how the CNN 'sees' the heartbeat trace.\n",
        "3.  **Low-Data Efficiency**: Demonstrating that even with a tiny dataset (100 samples), the AI learns the fundamental geometry of the ECG."
    ]
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and "def signal_to_image" in "".join(cell.get('source', [])):
            cell['source'] = [
                "def signal_to_image(signal, img_size=(128, 128), augment=False):\n",
                "    \"\"\"Converts a 1D signal into a 2D grayscale image with optional augmentation.\"\"\"\n",
                "    fig, ax = plt.subplots(figsize=(2, 2), dpi=64)\n",
                "    ax.plot(signal, color='black', linewidth=1.8)\n",
                "    ax.axis('off')\n",
                "    fig.canvas.draw()\n",
                "    \n",
                "    w, h = fig.canvas.get_width_height()\n",
                "    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)\n",
                "    img = img.reshape((h, w, 3))\n",
                "    plt.close(fig)\n",
                "    \n",
                "    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)\n",
                "    \n",
                "    if augment:\n",
                "        # Simulate real-world scanning issues\n",
                "        noise = np.random.normal(0, 10, img_gray.shape).astype(np.uint8)\n",
                "        img_gray = cv2.add(img_gray, noise)\n",
                "        angle = np.random.uniform(-3, 3)\n",
                "        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)\n",
                "        img_gray = cv2.warpAffine(img_gray, M, (w, h), borderValue=255)\n",
                "        \n",
                "    img_resized = cv2.resize(img_gray, img_size)\n",
                "    return img_resized / 255.0\n",
                "\n",
                "# 🎨 2.1 Transformation Pipeline Visualization\n",
                "print(\"🔍 Visualizing the 'Signal-to-Image' transformation pipeline...\")\n",
                "sample_sig = signals[0]\n",
                "clean_img = signal_to_image(sample_sig, augment=False)\n",
                "noisy_img = signal_to_image(sample_sig, augment=True)\n",
                "\n",
                "plt.figure(figsize=(15, 5), facecolor='#f8f9fa')\n",
                "plt.subplot(1, 3, 1)\n",
                "plt.plot(sample_sig, color='#34495e', linewidth=1)\n",
                "plt.title(\"1. Raw 1D Signal\", fontsize=12, fontweight='bold')\n",
                "plt.grid(True, alpha=0.3)\n",
                "\n",
                "plt.subplot(1, 3, 2)\n",
                "plt.imshow(clean_img, cmap='gray')\n",
                "plt.title(\"2. Synthetic 'Clean' ECG\", fontsize=12, fontweight='bold')\n",
                "plt.axis('off')\n",
                "\n",
                "plt.subplot(1, 3, 3)\n",
                "plt.imshow(noisy_img, cmap='gray')\n",
                "plt.title(\"3. Augmented (Real-World Sim)\", fontsize=12, fontweight='bold')\n",
                "plt.axis('off')\n",
                "\n",
                "plt.suptitle(\"Digitization Pipeline: Preparing the AI for Real-World Paper Records\", fontsize=16, y=1.05)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        if cell['cell_type'] == 'code' and "def visualize_internal_steps" in "".join(cell.get('source', [])):
            cell['source'] = [
                "def visualize_internal_steps(model, img_tensor):\n",
                "    model.eval()\n",
                "    activations = {}\n",
                "    \n",
                "    def get_activation(name):\n",
                "        def hook(model, input, output):\n",
                "            activations[name] = output.detach().cpu()\n",
                "        return hook\n",
                "    \n",
                "    # Attach hooks to visualize internal layers\n",
                "    h1 = model.features[0].register_forward_hook(get_activation('layer1'))\n",
                "    h2 = model.features[3].register_forward_hook(get_activation('layer2'))\n",
                "    \n",
                "    # Forward pass\n",
                "    with torch.no_grad():\n",
                "        model(img_tensor.to(device))\n",
                "    \n",
                "    h1.remove()\n",
                "    h2.remove()\n",
                "    \n",
                "    # Premium Plotting: Neural Pipeline\n",
                "    fig = plt.figure(figsize=(18, 10), facecolor='#fdfdfd')\n",
                "    gs = fig.add_gridspec(3, 8)\n",
                "    \n",
                "    # Top: Input\n",
                "    ax_in = fig.add_subplot(gs[0, 3:5])\n",
                "    ax_in.imshow(img_tensor[0, 0], cmap='gray')\n",
                "    ax_in.set_title(\"Step 1: Raw Image Input\", fontsize=14, fontweight='bold', pad=20)\n",
                "    ax_in.axis('off')\n",
                "    \n",
                "    # Middle: Layer 1 (Low-level Edge Detection)\n",
                "    act1 = activations['layer1'][0]\n",
                "    for i in range(min(8, act1.shape[0])):\n",
                "        ax = fig.add_subplot(gs[1, i])\n",
                "        ax.imshow(act1[i], cmap='magma')\n",
                "        if i == 0: ax.set_ylabel(\"Level 1: Edges\", fontsize=12, fontweight='bold')\n",
                "        ax.set_xticks([]); ax.set_yticks([])\n",
                "        ax.set_title(f\"Filter {i}\", fontsize=10)\n",
                "    \n",
                "    # Bottom: Layer 2 (Abstract Morphological Detection)\n",
                "    act2 = activations['layer2'][0]\n",
                "    for i in range(min(8, act2.shape[0])):\n",
                "        ax = fig.add_subplot(gs[2, i])\n",
                "        ax.imshow(act2[i], cmap='viridis')\n",
                "        if i == 0: ax.set_ylabel(\"Level 2: Concepts\", fontsize=12, fontweight='bold')\n",
                "        ax.set_xticks([]); ax.set_yticks([])\n",
                "        ax.set_title(f\"Filter {i}\", fontsize=10)\n",
                "        \n",
                "    plt.suptitle(\"Neural Network Internal Logic: From Visual Pixels to Diagnostic Features\", fontsize=18, y=1.02)\n",
                "    plt.tight_layout()\n",
                "    plt.show()\n",
                "\n",
                "test_img, test_sig = train_set[0]\n",
                "visualize_internal_steps(model, test_img.unsqueeze(0))"
            ]
    with open(path, 'w') as f: json.dump(nb, f, indent=1)

def update_signal_to_class():
    path = "/home/me/mini-proj/supervisor_notebooks/learning_signal_to_class.ipynb"
    if not os.path.exists(path): return
    with open(path, 'r') as f: nb = json.load(f)
    nb['cells'][0]['source'] = [
        "# 💓 Path 1: Signal-to-Class (1D ECG AI Classification)\n",
        "\n",
        "**Research Context**: This path explores the most direct form of AI diagnosis—processing raw electrical telemetry. We use 1D Convolutional Neural Networks to mimic a cardiologist's pattern recognition.\n",
        "\n",
        "### 🌟 Supervisor's Guide:\n",
        "1.  **Diagnostic Focus**: We overlay the AI's 'attention' (heatmap) on the raw signal to see which peaks determine the classification.\n",
        "2.  **Clinical Relevance**: Using real MIT-BIH data to distinguish between Normal, Ventricular, and Supraventricular beats.\n",
        "3.  **Model Transparency**: Seeing the internal 'filter response' that leads to a diagnosis."
    ]
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and "def visualize_1d_activations" in "".join(cell.get('source', [])):
            cell['source'] = [
                "def visualize_diagnostic_focus(model, sample_x, layer_idx=0):\n",
                "    activations = []\n",
                "    def hook(module, input, output):\n",
                "        activations.append(output.detach().cpu().numpy())\n",
                "    \n",
                "    handle = model.features[layer_idx].register_forward_hook(hook)\n",
                "    model(sample_x.to(device))\n",
                "    handle.remove()\n",
                "    \n",
                "    # Average activations across filters to see general 'attention'\n",
                "    act = np.mean(activations[0][0], axis=0)\n",
                "    act_normalized = (act - act.min()) / (act.max() - act.min() + 1e-8)\n",
                "    \n",
                "    signal = sample_x[0, 0].cpu().numpy()\n",
                "    \n",
                "    plt.figure(figsize=(15, 6), facecolor='#fdfdfd')\n",
                "    \n",
                "    # Plotting the signal with a heatmap overlay\n",
                "    plt.subplot(2, 1, 1)\n",
                "    plt.plot(signal, color='black', alpha=0.3, label='ECG Trace')\n",
                "    x_range = np.linspace(0, len(signal)-1, len(act))\n",
                "    plt.scatter(x_range, signal[x_range.astype(int)], c=act, cmap='Reds', s=50, alpha=0.8, label='AI Attention')\n",
                "    plt.title(\"1. Diagnostic Focus: Where the AI 'Looks' to Classify\", fontsize=14, fontweight='bold')\n",
                "    plt.legend()\n",
                "    \n",
                "    plt.subplot(2, 1, 2)\n",
                "    plt.imshow(activations[0][0], aspect='auto', cmap='magma')\n",
                "    plt.title(f\"2. Neural Filter Responses (Layer {layer_idx})\", fontsize=14, fontweight='bold')\n",
                "    plt.xlabel(\"Time Steps\")\n",
                "    plt.ylabel(\"Filter Index\")\n",
                "    plt.colorbar(label=\"Activation Strength\")\n",
                "    \n",
                "    plt.tight_layout()\n",
                "    plt.show()\n",
                "\n",
                "# Visualize for a class 2 (Ventricular) heartbeat\n",
                "v_indices = np.where(y_raw == 2)[0]\n",
                "target_idx = v_indices[0] if len(v_indices) > 0 else 0\n",
                "print(f\"🔍 Visualizing diagnostic focus for Class: {class_names[y_raw[target_idx]]}\")\n",
                "visualize_diagnostic_focus(model, X_tensor[target_idx:target_idx+1], layer_idx=0)"
            ]
    with open(path, 'w') as f: json.dump(nb, f, indent=1)

def update_image_to_class():
    path = "/home/me/mini-proj/supervisor_notebooks/learning_image_to_class.ipynb"
    if not os.path.exists(path): return
    with open(path, 'r') as f: nb = json.load(f)
    nb['cells'][0]['source'] = [
        "# 🖼️ Path 3: Image-to-Class (Direct Visual Diagnosis)\n",
        "\n",
        "**Research Context**: This path treats ECG analysis as a computer vision task. By transforming signals into images, we can leverage state-of-the-art CNN architectures to detect morphological anomalies.\n",
        "\n",
        "### 🌟 Supervisor's Guide:\n",
        "1.  **Shape Recognition**: See how the model distinguishes between different 'shapes' of heartbeats.\n",
        "2.  **Activation Heatmaps**: Visualizing the neural filters that detect specific curves and spikes.\n",
        "3.  **End-to-End Learning**: Demonstrating the model's ability to classify directly from pixels without manual feature engineering."
    ]
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and "def show_activations" in "".join(cell.get('source', [])):
            cell['source'] = [
                "def show_comparative_activations(model, img_tensor, indices):\n",
                "    model.eval()\n",
                "    fig, axes = plt.subplots(len(indices), 6, figsize=(18, 3*len(indices)), facecolor='#f8f9fa')\n",
                "    \n",
                "    for idx_row, sample_idx in enumerate(indices):\n",
                "        activation = {}\n",
                "        def get_activation(name):\n",
                "            def hook(model, input, output):\n",
                "                activation[name] = output.detach().cpu().numpy()\n",
                "            return hook\n",
                "        handle = model.features[0].register_forward_hook(get_activation('conv1'))\n",
                "        model(img_tensor[sample_idx:sample_idx+1])\n",
                "        handle.remove()\n",
                "        act = activation['conv1'][0]\n",
                "        axes[idx_row, 0].imshow(img_tensor[sample_idx, 0].cpu(), cmap='gray')\n",
                "        axes[idx_row, 0].set_title(f\"Input: {class_names[y_labels[sample_idx]]}\", fontsize=10, fontweight='bold')\n",
                "        axes[idx_row, 0].axis('off')\n",
                "        for i in range(5):\n",
                "            axes[idx_row, i+1].imshow(act[i], cmap='inferno')\n",
                "            axes[idx_row, i+1].set_title(f\"Filter {i+1}\", fontsize=9)\n",
                "            axes[idx_row, i+1].axis('off')\n",
                "    plt.suptitle(\"Comparative Neural Response: How the AI Sees Different Heartbeat Morphologies\", fontsize=18, y=1.02)\n",
                "    plt.tight_layout()\n",
                "    plt.show()\n",
                "n_idx = np.where(y_labels == 0)[0][0]\n",
                "v_idx = np.where(y_labels == 2)[0][0]\n",
                "show_comparative_activations(model, X_tensor, [n_idx, v_idx])"
            ]
    with open(path, 'w') as f: json.dump(nb, f, indent=1)

if __name__ == "__main__":
    update_image_to_signal()
    update_signal_to_class()
    update_image_to_class()
    print("✅ All 3 Supervisor Notebooks Refined Successfully")
