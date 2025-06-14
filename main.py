'''
COMS E6111: Adv Db Systems
Project 3
Authors: Anthony Flores-Alvarez
'''

# For reading and processing CSV files and tabular data using DataFrames
import pandas as pd
import itertools
import sys
from collections import defaultdict 

# Load dataset and convert each row into a set of 'column=value' items
def load_data(file_path):
    df = pd.read_csv(file_path, dtype=str).fillna('')

    # Forces column names as all 'item' for toy dataset
    if all(col.lower().startswith("item") for col in df.columns):
        df.columns = ['item'] * len(df.columns)

    transactions = []
    for _, row in df.iterrows():
        # Create a basket of items using the format 'column=value', ignoring blanks
        basket = [
            f"{col.strip().lower()}={str(item).strip().lower()}"
            for col, item in row.items()
            if str(item).strip() not in ["", "nan"]
        ]
        # frozenset() used for efficient lookup and counting
        transactions.append(frozenset(basket))
    return transactions

# Apply the Apriori algorithm to find frequent itemsets with support ≥ min_sup
def get_frequent_itemsets(transactions, min_sup):
    itemset_counts = defaultdict(int)
    total_transactions = len(transactions)

    # First pass: Count 1-itemsets
    for transaction in transactions:
        for item in transaction:
            itemset_counts[frozenset([item])] += 1

    def support(count): 
        return count / total_transactions

    frequent_itemsets = {}
    # Filter 1-itemsets by min_sup
    current_L = {itemset for itemset, count in itemset_counts.items() if support(count) >= min_sup}
    frequent_itemsets.update({itemset: support(itemset_counts[itemset]) for itemset in current_L})

    # Iteratively generate candidate itemsets of increasing size (k ≥ 2)
    k = 2
    while current_L:
        # SQL-Optimized Candidate Generation:
        # Only join itemsets that agree on their first (k-2) elements
        current_L_list = sorted([sorted(list(itemset)) for itemset in current_L])  # Sort items inside and outside
        candidate_sets = set()

        for i in range(len(current_L_list)):
            for j in range(i+1, len(current_L_list)):
                l1 = current_L_list[i]
                l2 = current_L_list[j]
                if l1[:k-2] == l2[:k-2]:  # Prefix match on first k-2 items
                    # Join them to form a k-itemset
                    candidate = frozenset(l1 + [l2[-1]])
                    candidate_sets.add(candidate)
                else:
                    break  # No further matches possible (because the list is sorted)

        candidate_sets = set(i.union(j) for i in current_L for j in current_L if len(i.union(j)) == k)
        # Prune: only keep candidates whose all (k-1)-subsets are frequent
        pruned_candidates = {
            c for c in candidate_sets if all(
                frozenset(sub) in current_L for sub in itertools.combinations(c, k-1)
            )
        }

        # Count support for remaining candidate itemsets
        itemset_counts = defaultdict(int)
        for transaction in transactions:
            for candidate in pruned_candidates:
                if candidate.issubset(transaction):
                    itemset_counts[candidate] += 1

        # Filter by min_sup and continue loop
        current_L = {itemset for itemset, count in itemset_counts.items() if support(count) >= min_sup}
        frequent_itemsets.update({itemset: support(itemset_counts[itemset]) for itemset in current_L})
        k += 1

    return frequent_itemsets, total_transactions

# Generate association rules with confidence ≥ min_conf and lift ≥ min_lift
# Lift used to filter out rules that aren't very (statistically) 'interesting'
# Idea from here: 
# - https://stackoverflow.com/questions/2008488/minimum-confidence-and-minimum-support-for-apriori
# - https://stackoverflow.com/questions/46495018/association-rule-mining-confidence-and-lift
def generate_rules(frequent_itemsets, min_conf, total_transactions, min_lift=2.0):
    rules = []
    seen_signatures = set()  # To avoid adding duplicate or inverse rules

    # Pairs of fields considered redundant (e.g., location synonyms)
    redundant_pairs = {
        ('violation location (borough)', 'borough'),
        ('violation location (zip code)', 'zip_code'),
        ('violation location (city)', 'borough'),
        ('violation location (zip code)', 'borough'),
        ('zip_code', 'borough'),
        ('zip_code', 'violation location (borough)'),
        ('violation location (zip code)', 'violation location (city)'),
        ('zip_code', 'violation location (city)'),
        ('violation location (city)', 'violation location (borough)'),
        ('violation location (zip code)', 'violation location (borough)'),
        ('violation location (borough)', 'violation location (city)'),
        ('borough', 'violation location (borough)'),
        ('borough', 'violation location (city)'),
        ('borough', 'violation location (zip_code)')
    }

    # Check if a rule is trivially redundant due to equivalent attributes
    def is_trivially_related(lhs, rhs):
        lhs_attrs = {i.split('=')[0].strip() for i in lhs}
        rhs_attrs = {i.split('=')[0].strip() for i in rhs}
        for a, b in redundant_pairs:
            if (a in lhs_attrs and b in rhs_attrs) or (b in lhs_attrs and a in rhs_attrs):
                return True
        return False

    # Iterate through all frequent itemsets to build valid rules
    for itemset in frequent_itemsets:
        if len(itemset) < 2:
            continue
        for i in range(1, len(itemset)):
            for lhs in itertools.combinations(itemset, i):
                lhs = frozenset(lhs)
                rhs = itemset - lhs
                if not rhs:
                    continue

                # Skip if reverse rule already seen or trivial redundancy detected
                key = (tuple(sorted(lhs)), tuple(sorted(rhs)))
                reverse_key = (key[1], key[0])
                if reverse_key in seen_signatures or key in seen_signatures:
                    continue
                if is_trivially_related(lhs, rhs):
                    continue

                support_lhs = frequent_itemsets.get(lhs, 0)
                support_rhs = frequent_itemsets.get(rhs, 0)
                support_both = frequent_itemsets[itemset]

                if support_lhs > 0 and support_rhs > 0:
                    confidence = support_both / support_lhs
                    lift = confidence / support_rhs

                    # Add rule only if it passes min thresholds
                    if confidence >= min_conf and lift >= min_lift:
                        rules.append((lhs, rhs, support_both, confidence, lift))
                        seen_signatures.add(key)
    return rules

# Write frequent itemsets and rules to output.txt (includes custom scoring + tagging)
def format_output(frequent_itemsets, rules, output_file):
    # Field origins and geo-tags for rule interpretation
    oath_prefixes = {
        'violation date', 'issuing agency', 'violation location (borough)', 'violation location (city)',
        'violation location (zip code)', 'hearing status', 'hearing result',
        'compliance status', 'charge #1: code description'
    }
    rodent_prefixes = {
        'inspection_type', 'zip_code', 'borough', 'result', 'approved_date'
    }
    geo_keywords = {'zip_code', 'borough', 'violation location (zip code)', 'violation location (borough)', 'violation location (city)'}

    # Determine dataset origin of a field
    def item_origin(item):
        key = item.split('=')[0].strip()
        if key in oath_prefixes:
            return 'OATH'
        elif key in rodent_prefixes:
            return 'RODENT'
        else:
            return 'UNKNOWN'

    # Cross-dataset: items are from both OATH and Rodent datasets
    def is_cross_dataset(lhs, rhs):
        sources = {item_origin(item) for item in lhs.union(rhs)}
        return 'OATH' in sources and 'RODENT' in sources

    # Geo-linked: LHS or RHS contains geographic info
    def is_geo_linked(lhs, rhs):
        return any(item.split('=')[0].strip() in geo_keywords for item in lhs.union(rhs))

    # Score rules based on tags and 'interestingness' (subjective lol)
    def interestingness_score(lhs, rhs, conf, lift):
        score = 0
        if is_cross_dataset(lhs, rhs):
            score += 2
        if is_geo_linked(lhs, rhs):
            score += 1
        score += lift
        score += conf
        return score

    # Write frequent itemsets sorted by support
    with open(output_file, 'w') as f:
        f.write("==Frequent itemsets==\n")
        for itemset, support in sorted(frequent_itemsets.items(), key=lambda x: -x[1]):
            f.write(f"[{','.join(sorted(itemset))}], {support*100:.4f}%\n")

        # Write rules sorted by confidence
        f.write("\n==High-confidence association rules==\n")
        for lhs, rhs, support, confidence, lift in sorted(rules, key=lambda x: -x[3]):
            tags = []
            if is_cross_dataset(lhs, rhs):
                tags.append("CROSS-DATASET")
            if is_geo_linked(lhs, rhs):
                tags.append("GEO-LINKED")
            tag_str = f" [{' | '.join(tags)}]" if tags else ""
            f.write(
                f"[{','.join(sorted(lhs))}] => [{','.join(sorted(rhs))}] "
                f"(Conf: {confidence*100:.1f}%, Supp: {support*100:.4f}%, Lift: {lift:.2f}){tag_str}\n"
            )

        # Write rules sorted by our custom interestingness score
        f.write("\n==Ranked by Interestingness==\n")
        sorted_by_score = sorted(rules, key=lambda x: -interestingness_score(x[0], x[1], x[3], x[4]))
        for lhs, rhs, support, confidence, lift in sorted_by_score:
            score = interestingness_score(lhs, rhs, confidence, lift)
            tags = []
            if is_cross_dataset(lhs, rhs):
                tags.append("CROSS-DATASET")
            if is_geo_linked(lhs, rhs):
                tags.append("GEO-LINKED")
            tag_str = f" [{' | '.join(tags)}]" if tags else ""
            f.write(
                f"[{','.join(sorted(lhs))}] => [{','.join(sorted(rhs))}] "
                f"(Score: {score:.2f}, Conf: {confidence*100:.1f}%, Supp: {support*100:.4f}%, Lift: {lift:.2f}){tag_str}\n"
            )

# Entry point (MAIN)
def main():
    if len(sys.argv) != 4:
        print("Usage: python3 main.py <dataset.csv> <min_sup> <min_conf>")
        sys.exit(1)

    file_path = sys.argv[1]
    min_sup = float(sys.argv[2])
    min_conf = float(sys.argv[3])

    transactions = load_data(file_path)
    frequent_itemsets, total = get_frequent_itemsets(transactions, min_sup)
    # for toy example, set this to 1.0
    rules = generate_rules(frequent_itemsets, min_conf, total, min_lift=1.0)
    format_output(frequent_itemsets, rules, "output.txt")

if __name__ == "__main__":
    main()
