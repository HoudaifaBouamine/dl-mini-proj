# Run from the repository root (paths below are relative to it).
import json
import os

def update_class_explanations():
    files = [
        "notebooks/supervisor/learning_signal_to_class.ipynb",
        "notebooks/supervisor/learning_image_to_class.ipynb"
    ]
    
    explanation_md = [
        "### 🏥 Note on 'Unknown' Class (Category Q):\n",
        "The **Unknown** class is a standard clinical category in the MIT-BIH dataset. It includes:\n",
        "- **Paced beats** (from artificial pacemakers)\n",
        "- **Unclassifiable signals** (distorted by noise or artifacts)\n",
        "\n",
        "In a real-world deployment, 'Unknown' triggers a **manual review** by a cardiologist, ensuring AI safety."
    ]

    for path in files:
        if not os.path.exists(path): continue
        with open(path, 'r') as f: nb = json.load(f)
        
        # Update class names in code cells
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = "".join(cell.get('source', []))
                if 'class_names =' in source:
                    cell['source'] = [s.replace('["Normal", "SVEB", "VEB", "Fusion", "Unknown"]', '["Normal", "SVEB", "VEB", "Fusion", "Unknown/Paced"]')
                                     .replace('["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]', '["Normal", "SVEB", "VEB", "Fusion", "Unknown/Paced"]')
                                     for s in cell['source']]
        
        # Add explanation markdown before the result/viz section
        insert_idx = -1
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'markdown' and "Internal Mechanism" in "".join(cell.get('source', [])):
                insert_idx = i
                break
        
        if insert_idx != -1:
            new_cell = {"cell_type": "markdown", "metadata": {}, "source": explanation_md}
            nb['cells'].insert(insert_idx, new_cell)
            
        with open(path, 'w') as f: json.dump(nb, f, indent=1)

if __name__ == "__main__":
    update_class_explanations()
    print("✅ Educational notes on 'Unknown' class added to notebooks.")
