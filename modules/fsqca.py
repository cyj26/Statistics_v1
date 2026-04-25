# =============================================================================
# fsQCA (Fuzzy-Set Qualitative Comparative Analysis) 모듈
# 직접 구현: 보정(calibration) → 진리표(truth table) → 부울 최소화
# 참고: Ragin (2008), Redesigning Social Inquiry
# =============================================================================
import pandas as pd
import numpy as np
from itertools import combinations, chain


# ── 1. 보정 (Calibration) ─────────────────────────────────────────────────────
def calibrate_direct(series: pd.Series, full_in: float,
                     crossover: float, full_out: float) -> pd.Series:
    """직접 보정법 (Ragin 3점 기준)"""
    s = series.copy().astype(float)
    result = pd.Series(index=s.index, dtype=float)
    for i, val in s.items():
        if val >= full_in:
            result[i] = 0.99
        elif val <= full_out:
            result[i] = 0.01
        else:
            # 로지스틱 변환
            log_odds = np.log((val - full_out + 1e-9) / (full_in - val + 1e-9))
            result[i] = float(1 / (1 + np.exp(-log_odds)))
    return result.clip(0.01, 0.99)


# ── 2. 필요조건 분석 ──────────────────────────────────────────────────────────
def necessary_conditions(df_fs: pd.DataFrame, outcome: str,
                          conditions: list, threshold: float = 0.9):
    rows = []
    y = df_fs[outcome]
    for cond in conditions:
        x = df_fs[cond]
        cov = float((x * y).sum() / (y.sum() + 1e-9))
        cons = float((x * y).sum() / (x.sum() + 1e-9))
        rows.append({
            "조건": cond,
            "일관성(Consistency)": round(cons, 3),
            "포함도(Coverage)":    round(cov, 3),
            "필요조건":  "✓" if cons >= threshold else "✗"
        })
    return pd.DataFrame(rows)


# ── 3. 진리표 구성 ────────────────────────────────────────────────────────────
def build_truth_table(df_fs: pd.DataFrame, outcome: str,
                       conditions: list, freq_threshold: int = 1,
                       cons_threshold: float = 0.75):
    n_conds = len(conditions)
    rows = []

    for combo in range(2 ** n_conds):
        config = [(combo >> i) & 1 for i in range(n_conds - 1, -1, -1)]
        mask = pd.Series([True] * len(df_fs), index=df_fs.index)
        membership = pd.Series([1.0] * len(df_fs), index=df_fs.index)

        for ci, (cond, val) in enumerate(zip(conditions, config)):
            if val == 1:
                membership = membership * df_fs[cond]
            else:
                membership = membership * (1 - df_fs[cond])

        row_members = membership[membership >= 0.5]
        freq = len(row_members)
        if freq < freq_threshold:
            continue

        y_vals = df_fs.loc[row_members.index, outcome]
        m_vals = row_members

        cons = float((m_vals * y_vals).sum() / (m_vals.sum() + 1e-9))
        cov  = float((m_vals * y_vals).sum() / (df_fs[outcome].sum() + 1e-9))

        row = {}
        for ci, cond in enumerate(conditions):
            row[cond] = config[ci]
        row["빈도(N)"]            = freq
        row["일관성(Consistency)"] = round(cons, 3)
        row["포함도(Coverage)"]    = round(cov, 3)
        row["결과(1=포함)"]        = 1 if cons >= cons_threshold else 0
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── 4. 충분조건 분석 (단순 버전) ──────────────────────────────────────────────
def sufficient_conditions(truth_table: pd.DataFrame, outcome: str,
                           conditions: list, cons_threshold: float = 0.75):
    """진리표에서 일관성 충족 행 추출 → 충분조건 패턴 반환"""
    if truth_table.empty: return pd.DataFrame()
    sufficient = truth_table[truth_table["결과(1=포함)"] == 1].copy()
    if sufficient.empty: return pd.DataFrame()

    result_rows = []
    for _, row in sufficient.iterrows():
        parts = []
        for cond in conditions:
            val = row[cond]
            parts.append(f"{'~' if val==0 else ''}{cond}")
        result_rows.append({
            "충분조건 조합": " * ".join(parts),
            "일관성": row["일관성(Consistency)"],
            "포함도": row["포함도(Coverage)"],
            "빈도":   row["빈도(N)"]
        })
    return pd.DataFrame(result_rows)


# ── 5. 전체 fsQCA 실행 ────────────────────────────────────────────────────────
def run_fsqca(df: pd.DataFrame, outcome_col: str, condition_cols: list,
              calibration_params: dict,   # {col: (full_in, crossover, full_out)}
              freq_threshold: int = 1,
              cons_threshold: float = 0.75,
              nec_threshold:  float = 0.9):
    """
    Returns: dict with keys = 분석단계 이름, values = DataFrame
    """
    # 보정
    df_fs = pd.DataFrame(index=df.index)
    calib_info = []
    for col in [outcome_col] + condition_cols:
        if col in calibration_params:
            fi, co, fo = calibration_params[col]
            df_fs[col] = calibrate_direct(df[col], fi, co, fo)
            calib_info.append({"변수": col, "완전포함(1)": fi,
                                "교차점(.5)": co, "완전배제(0)": fo})
        else:
            # 자동 보정: 5%, 50%, 95% 분위
            q = df[col].quantile([0.05, 0.5, 0.95])
            df_fs[col] = calibrate_direct(df[col], q[0.95], q[0.5], q[0.05])
            calib_info.append({"변수": col, "완전포함(1)": round(q[0.95],2),
                                "교차점(.5)": round(q[0.5],2), "완전배제(0)": round(q[0.05],2)})

    calib_df = pd.DataFrame(calib_info)

    # 기술통계 (보정 후)
    desc_fs = df_fs.describe().T[["mean","std","min","max"]].round(3)
    desc_fs.columns = ["평균","표준편차","최솟값","최댓값"]
    desc_fs = desc_fs.reset_index().rename(columns={"index":"변수"})

    # 필요조건
    nec_df = necessary_conditions(df_fs, outcome_col, condition_cols, nec_threshold)

    # 진리표
    tt = build_truth_table(df_fs, outcome_col, condition_cols,
                            freq_threshold, cons_threshold)

    # 충분조건
    suf_df = sufficient_conditions(tt, outcome_col, condition_cols, cons_threshold)

    # 전체 해 통계
    if not suf_df.empty:
        sol_cons = suf_df["일관성"].mean()
        sol_cov  = suf_df["포함도"].mean()
        sol_summary = pd.DataFrame([{
            "해 수(충분조건 조합)": len(suf_df),
            "평균 일관성": round(sol_cons, 3),
            "평균 포함도": round(sol_cov, 3),
            "분석 기준(일관성 임계값)": cons_threshold
        }])
    else:
        sol_summary = pd.DataFrame([{"안내": "일관성 기준을 충족하는 충분조건 조합이 없습니다."}])

    return {
        "보정기준":    calib_df,
        "보정후기술통계": desc_fs,
        "필요조건분석": nec_df,
        "진리표":      tt,
        "충분조건분석": suf_df,
        "해요약":      sol_summary
    }
