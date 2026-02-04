#!/usr/bin/env python3
"""
Summary script to display key findings from the optimizer comparison.
"""

import json
import pandas as pd

def main():
    # Load results
    with open('optimizer_comparison_results.json', 'r') as f:
        results = json.load(f)
    
    print("🎯 OPTIMIZER COMPARISON - KEY FINDINGS")
    print("=" * 60)
    
    # Summary table
    summary_data = []
    for dataset_name, dataset_results in results.items():
        for opt_name, opt_results in dataset_results.items():
            if 'error' not in opt_results:
                summary_data.append({
                    'Dataset': dataset_name.replace('_', ' ').title(),
                    'Optimizer': opt_name.upper(),
                    'R² Score': f"{opt_results['test_r2']:.3f}",
                    'MSE': f"{opt_results['test_mse']:.3f}",
                    'Time (s)': f"{opt_results['total_time']:.1f}",
                    'Epochs': opt_results['epochs_trained']
                })
    
    df = pd.DataFrame(summary_data)
    print("\n📊 PERFORMANCE SUMMARY:")
    print(df.to_string(index=False))
    
    print("\n🏆 KEY INSIGHTS:")
    print("- AdamW consistently outperformed other optimizers")
    print("- NovoGrad showed reasonable performance but was less stable")
    print("- Shampoo struggled with the learning rate tuning")
    print("- Muon had distributed computing requirements in this setup")
    
    print("\n📈 PRACTICAL RECOMMENDATIONS:")
    print("1. Use AdamW as the default optimizer for tabular data")
    print("2. Consider NovoGrad for deep architectures with many layers")
    print("3. Tune learning rates carefully for second-order methods like Shampoo")
    print("4. AdamW provides the best balance of performance and stability")
    
    print(f"\n📁 Generated files:")
    print(f"- tutorials/05_optimizers_comparison.ipynb")
    print(f"- models/optimizers.py") 
    print(f"- optimizers_comparison.png")
    print(f"- optimizers_efficiency.png")
    print(f"- optimizer_comparison_results.json")

if __name__ == "__main__":
    main()