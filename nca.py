# =============================================================================
# NCA (Necessary Condition Analysis) 모듈
# CE-FDH (Ceiling Envelopment - Free Disposal Hull) 방식 직접 구현
# 참고: Dul (2016), Journal of Business Logistics
# =============================================================================
import pandas as pd
import numpy as np


def _ce_fdh(x: np.ndarray, y: np.ndarray):
    """CE-FDH 천장선: 각 x에 대해 가능한 최대 y를 반환"""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    ceiling_x, ceiling_y = [xs[0]], [ys[0]]
    for i in range(1, len(xs)):
        if ys[i] > ceiling_y[-1]:
            ceiling_x.append(xs[i])
            ceiling_y.append(ys[i])
    return np.array(ceiling_x), np.array(ceiling_y)


def _cr_fdh(x: np.ndarray, y: np.ndarray):
    """CR-FDH: CE-FDH 천장점에 OLS 적합"""
    cx, cy = _ce_fdh(x, y)
    if len(cx) < 2:
        return None, None, None
    coeffs = np.polyfit(cx, cy, 1)
    slope, intercept = coeffs
    y_pred = slope * cx + intercept
    ss_res = np.sum((cy - y_pred)**2)
    ss_tot = np.sum((cy - np.mean(cy))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slope, intercept, r2


def _effect_size(x, y, slope, intercept, method="cr_fdh"):
    """NCA 효과 크기 d = 천장 영역 / 전체 범위"""
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    scope   = x_range * y_range
    if scope == 0: return 0.0

    # 천장선 위쪽 면적 (조건 공간) — NumPy 1.x: trapz / 2.x: trapezoid
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    xs = np.linspace(x.min(), x.max(), 500)
    y_ceiling = np.clip(slope * xs + intercept, y.min(), y.max())
    ceiling_area = _trapz(y_ceiling - y.min(), xs)
    d = ceiling_area / scope
    return round(float(np.clip(d, 0, 1)), 4)


def run_nca(df: pd.DataFrame, x_cols: list, y_col: str,
            p_threshold: float = 0.05):
    """
    Parameters
    ----------
    x_cols : 필요조건 후보 독립변수 목록
    y_col  : 결과변수
    """
    data = df[[y_col] + x_cols].dropna()
    y = data[y_col].values
    results = []
    ceiling_data = {}

    for xcol in x_cols:
        x = data[xcol].values
        slope, intercept, r2_ceil = _cr_fdh(x, y)
        if slope is None:
            results.append({"변수": xcol, "효과크기(d)": np.nan,
                             "CR-FDH 기울기": np.nan, "절편": np.nan,
                             "R²(ceiling)": np.nan, "해석": "데이터 부족"})
            continue

        d = _effect_size(x, y, slope, intercept)

        # 효과크기 해석 (Dul 2016 기준)
        if d < 0.1:   interp = "매우 작음"
        elif d < 0.3: interp = "작음"
        elif d < 0.5: interp = "중간"
        else:          interp = "큼"

        results.append({
            "변수(X)":       xcol,
            "결과변수(Y)":   y_col,
            "효과크기(d)":   d,
            "CR-FDH 기울기": round(slope, 4),
            "절편":          round(intercept, 4),
            "R²(ceiling)":   round(r2_ceil, 3) if r2_ceil is not None else np.nan,
            "해석":          interp,
            "필요조건 판단": "필요조건 ✓" if d >= 0.1 else "필요조건 아님"
        })

        # 병목 분석 데이터 저장
        xs_bn = np.arange(0, 110, 10)  # 0~100% 수준
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        bn_rows = []
        for xpct in xs_bn:
            x_val = x_min + xpct / 100 * (x_max - x_min)
            y_val = slope * x_val + intercept
            y_pct = (y_val - y_min) / (y_max - y_min) * 100 if y_max > y_min else np.nan
            bn_rows.append({"X(%)": xpct, f"{xcol}_최소필요Y(%)": round(float(np.clip(y_pct, 0, 100)), 1)})
        ceiling_data[xcol] = pd.DataFrame(bn_rows)

    result_df = pd.DataFrame(results)

    # 병목 테이블 통합
    if ceiling_data:
        bottleneck = ceiling_data[x_cols[0]][["X(%)"]]
        for xcol, bdf in ceiling_data.items():
            bottleneck = bottleneck.merge(bdf, on="X(%)")
    else:
        bottleneck = pd.DataFrame()

    return result_df, bottleneck
