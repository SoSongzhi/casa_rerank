#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze modifications in the database
"""

import sys
import re
import io
from collections import Counter

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def extract_modifications(peptide):
    """Extract all modifications from a peptide sequence"""
    # Match both () and [] formats
    pattern = r'[\[\(]([+\-]?\d*\.?\d+)[\]\)]'
    mods = re.findall(pattern, peptide)
    return [float(m) for m in mods if m]

def classify_modification(mass):
    """Classify modification by mass"""
    mass_ranges = [
        ((0.83, 1.13), '+0.98', 'Deamidation (N/Q)', 'UNIMOD:7'),
        ((15.85, 16.15), '+15.99', 'Oxidation (M)', 'UNIMOD:35'),
        ((56.87, 57.17), '+57.02', 'Carbamidomethyl (C)', 'UNIMOD:4'),
        ((41.86, 42.16), '+42.01', 'Acetyl (N-term)', 'UNIMOD:1'),
        ((42.86, 43.16), '+43.01', 'Carbamyl (N-term)', 'UNIMOD:5'),
        ((-17.18, -16.88), '-17.03', 'Ammonia-loss', 'UNIMOD:385'),
        ((-18.16, -17.86), '-18.01', 'Glu->pyro-Glu', 'UNIMOD:27'),
        ((25.83, 26.13), '+25.98', 'Carbamyl+Ammonia', 'UNIMOD:526'),
        ((79.96, 80.06), '+80.00', 'Phosphorylation', 'UNIMOD:21'),
        ((14.01, 14.03), '+14.02', 'Methylation', 'UNIMOD:34'),
    ]
    
    for (min_m, max_m), std_mass, name, unimod in mass_ranges:
        if min_m <= mass <= max_m:
            return std_mass, name, unimod
    
    # Unknown modification
    if mass >= 0:
        return f'+{mass:.2f}', 'Unknown', 'N/A'
    else:
        return f'{mass:.2f}', 'Unknown', 'N/A'

def analyze_database(mgf_file):
    """Analyze all modifications in the database"""
    
    print("="*80)
    print("Database Modification Analysis")
    print("="*80)
    print(f"Analyzing: {mgf_file}\n")
    
    total_peptides = 0
    peptides_with_mods = 0
    all_modifications = []
    mod_counter = Counter()
    peptide_examples = {}
    
    try:
        with open(mgf_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('SEQ='):
                    total_peptides += 1
                    seq = line.strip().split('=', 1)[1]
                    
                    # Extract modifications
                    mods = extract_modifications(seq)
                    
                    if mods:
                        peptides_with_mods += 1
                        all_modifications.extend(mods)
                        
                        # Count each modification
                        for mod in mods:
                            std_mass, name, unimod = classify_modification(mod)
                            mod_key = f"{std_mass} ({name})"
                            mod_counter[mod_key] += 1
                            
                            # Store example peptide
                            if mod_key not in peptide_examples:
                                peptide_examples[mod_key] = seq
    
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Print statistics
    print(f"Total peptides in database: {total_peptides:,}")
    print(f"Peptides with modifications: {peptides_with_mods:,}")
    print(f"Percentage with modifications: {peptides_with_mods/total_peptides*100:.2f}%")
    print(f"Total modification instances: {len(all_modifications):,}")
    print()
    
    # Print modification types and frequencies
    print("="*80)
    print("Modification Types and Frequencies")
    print("="*80)
    print(f"{'Rank':<6} {'Modification':<35} {'Count':<10} {'%':<8} {'Unimod'}")
    print("-"*80)
    
    for rank, (mod_key, count) in enumerate(mod_counter.most_common(), 1):
        percentage = count / len(all_modifications) * 100
        
        # Extract Unimod ID
        std_mass = mod_key.split(' ')[0]
        for mass in set(all_modifications):
            s, n, u = classify_modification(mass)
            if s == std_mass:
                unimod_id = u
                break
        else:
            unimod_id = 'N/A'
        
        print(f"{rank:<6} {mod_key:<35} {count:<10,} {percentage:>6.2f}%  {unimod_id}")
    
    print("-"*80)
    print(f"{'Total':<6} {'':<35} {len(all_modifications):<10,} {100.0:>6.2f}%")
    print()
    
    # Print examples
    print("="*80)
    print("Example Peptides for Each Modification")
    print("="*80)
    
    for mod_key in mod_counter.most_common():
        mod_name = mod_key[0]
        example = peptide_examples.get(mod_name, 'N/A')
        print(f"\n{mod_name}:")
        print(f"  Example: {example}")
    
    print("\n" + "="*80)
    
    # Detailed mass distribution
    print("\nDetailed Mass Distribution:")
    print("-"*80)
    
    mass_counter = Counter(all_modifications)
    for mass, count in sorted(mass_counter.items()):
        std_mass, name, unimod = classify_modification(mass)
        print(f"  {mass:>8.2f} Da -> {std_mass:>8} ({name:<30}) Count: {count:>5,}  {unimod}")
    
    print("="*80)

if __name__ == "__main__":
    database_file = "test_data/high_nine/high_nine_database.mgf"
    analyze_database(database_file)