#!/usr/bin/env python3
"""
Standalone Formal Verification Test

Runs formal verification without importing the full VLA pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

# Direct import
exec(open(os.path.join(os.path.dirname(__file__), 'semantic_parser.py')).read())
exec(open(os.path.join(os.path.dirname(__file__), 'formal_verification.py')).read().replace('from .semantic_parser', '# from .semantic_parser'))

if __name__ == "__main__":
    print("="*80)
    print("FORMAL VERIFICATION OF LANGUAGE → ACTION SEMANTICS")
    print("="*80)
    
    report = run_formal_verification()
    
    # Print summary
    print("\n## GRAMMAR SPECIFICATION")
    print(f"  Terminals: {report['grammar']['terminals']}")
    print(f"  Non-terminals: {report['grammar']['non_terminals']}")
    print(f"  Production rules: {report['grammar']['production_rules']}")
    
    print("\n## CORRECTNESS PROPERTIES")
    print(f"  Completeness rate: {report['correctness']['completeness']['completeness_rate']:.1%}")
    print(f"  Soundness rate: {report['correctness']['soundness']['soundness_rate']:.1%}")
    print(f"  Determinism rate: {report['correctness']['determinism']['determinism_rate']:.1%}")
    print(f"  Ambiguity rate: {report['correctness']['ambiguity']['ambiguity_rate']:.1%}")
    
    print("\n## COVERAGE ANALYSIS")
    spatial_cov = report['coverage']['spatial_relations']
    print(f"  Spatial relations coverage: {spatial_cov['coverage_percentage']:.1f}%")
    print(f"  Supported: {spatial_cov['supported']}/{spatial_cov['total_possible']}")
    
    ling_cov = report['coverage']['linguistic_constructs']
    print(f"  Linguistic constructs support rate: {ling_cov['support_rate']:.1%}")
    
    print("\n## ADVERSARIAL TESTING")
    adv = report['adversarial']['adversarial_robustness']
    print(f"  Rejection rate: {adv['rejection_rate']:.1%}")
    print(f"  Correctly rejected: {adv['correctly_rejected']}/{adv['total']}")
    print(f"  Incorrectly accepted: {adv['incorrectly_accepted']}")
    
    sem_eq = report['adversarial']['semantic_equivalence']
    print(f"  Paraphrase equivalence: {sem_eq['equivalence_rate']:.1%}")
    
    print("\n## CONFUSION MATRIX (Spatial Relations)")
    cm = report['adversarial']['confusion_matrix']
    for true_rel, preds in cm.items():
        print(f"  {true_rel}: {dict(preds)}")
    
    # Save full report
    output_file = 'formal_verification_report.json'
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Full report saved to: {output_file}")
    print("="*80)
