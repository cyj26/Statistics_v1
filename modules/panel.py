import pandas as pd
import numpy as np
from modules.utils import fmt_p, sig_stars


def run_panel(df, entity_col, time_col, dep_col, indep_cols):
    try:
        from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
    except ImportError:
        return None, "linearmodels 패키지가 필요합니다: pip install linearmodels"

    try:
        data = df[[entity_col, time_col, dep_col] + indep_cols].dropna().copy()

        # 시간 변수 변환
        try:
            data[time_col] = pd.to_datetime(data[time_col])
        except Exception:
            data[time_col] = pd.to_numeric(data[time_col], errors="coerce")

        data = data.set_index([entity_col, time_col])
        dep  = data[dep_col]
        exog = data[indep_cols]

        # 모형 추정
        fe = PanelOLS(dep, exog, entity_effects=True).fit(
            cov_type="clustered", cluster_entity=True)
        re = RandomEffects(dep, exog).fit()
        po = PooledOLS(dep, exog).fit()

        def extract(res, label):
            rows = []
            for var in indep_cols:
                try:
                    rows.append({
                        "변수": var, "모형": label,
                        "계수(β)":  round(float(res.params[var]), 4),
                        "표준오차": round(float(res.std_errors[var]), 4),
                        "t/z값":    round(float(res.tstats[var]), 3),
                        "p값":      fmt_p(float(res.pvalues[var])),
                        "유의성":   sig_stars(float(res.pvalues[var])),
                    })
                except Exception:
                    pass
            return pd.DataFrame(rows)

        coef_all = pd.concat([extract(fe,"고정효과(FE)"),
                               extract(re,"확률효과(RE)"),
                               extract(po,"Pooled OLS")], ignore_index=True)

        # Hausman 검정
        try:
            from linearmodels.panel import hausman
            h_stat, h_p = hausman(fe, re)
            hausman_df = pd.DataFrame([{
                "검정통계량": round(float(h_stat), 3),
                "p값": fmt_p(float(h_p)),
                "결론": "고정효과 모형 선택 권장 (p<.05)" if float(h_p) < 0.05
                        else "확률효과 모형 선택 가능 (p≥.05)"
            }])
        except Exception:
            hausman_df = pd.DataFrame([{"안내": "Hausman 검정 불가 (linearmodels 버전 확인 필요)"}])

        # 적합도
        def safe_r2(res):
            try: return round(float(res.rsquared), 3)
            except: return "-"

        fit_df = pd.DataFrame([
            {"모형": "고정효과(FE)", "R²": safe_r2(fe), "관측수": int(fe.nobs)},
            {"모형": "확률효과(RE)", "R²": safe_r2(re), "관측수": int(re.nobs)},
            {"모형": "Pooled OLS",   "R²": safe_r2(po), "관측수": int(po.nobs)},
        ])

        return {"계수비교": coef_all, "모형적합도": fit_df, "Hausman검정": hausman_df}, None

    except Exception as e:
        return None, f"패널분석 오류: {str(e)}"
