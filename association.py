import pandas as pd
import numpy as np


def run_association(df, item_cols, min_support=0.05,
                    min_confidence=0.3, min_lift=1.0, max_len=3):
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
    except ImportError:
        return None, None, "mlxtend 패키지가 필요합니다: pip install mlxtend"

    try:
        data = df[item_cols].copy()

        # 이진 변환
        flat = data.values.flatten()
        flat_clean = flat[~pd.isna(flat)]
        unique_vals = set(flat_clean)

        if unique_vals <= {0, 1, True, False, 0.0, 1.0}:
            basket = data.fillna(0).astype(bool)
        else:
            basket = pd.get_dummies(data.astype(str)).astype(bool)

        # Apriori
        freq_items = apriori(basket, min_support=min_support,
                             use_colnames=True, max_len=max_len)

        if freq_items.empty:
            return pd.DataFrame(), pd.DataFrame(), \
                   f"지지도 {min_support} 이상인 빈발항목이 없습니다. 임계값을 낮춰보세요."

        freq_items["항목집합"] = freq_items["itemsets"].apply(
            lambda x: ", ".join(sorted(list(x))))
        freq_out = freq_items[["항목집합","support"]].copy()
        freq_out.columns = ["항목집합","지지도"]
        freq_out = freq_out.sort_values("지지도", ascending=False).reset_index(drop=True)
        freq_out["지지도"] = freq_out["지지도"].round(4)

        # 연관 규칙
        rules = association_rules(freq_items, metric="confidence",
                                  min_threshold=min_confidence,
                                  num_itemsets=len(freq_items))
        if rules.empty:
            return freq_out, pd.DataFrame(), \
                   f"신뢰도 {min_confidence} 이상인 규칙이 없습니다."

        rules = rules[rules["lift"] >= min_lift].copy()
        rules["조건부(IF)"]  = rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
        rules["결과(THEN)"] = rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))

        rules_out = rules[["조건부(IF)","결과(THEN)",
                            "support","confidence","lift"]].copy()
        rules_out.columns = ["조건부(IF)","결과(THEN)","지지도","신뢰도","향상도"]
        rules_out = rules_out.sort_values("향상도", ascending=False).reset_index(drop=True)
        for col in ["지지도","신뢰도","향상도"]:
            rules_out[col] = rules_out[col].round(4)

        return freq_out, rules_out, None

    except Exception as e:
        return None, None, f"연관분석 오류: {str(e)}"
