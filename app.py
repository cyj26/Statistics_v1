# =============================================================================
# 📊 통계 자동 분석기
# 실행: python -m streamlit run app.py
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import defaultdict

st.set_page_config(page_title="통계 자동 분석기",
                   page_icon="📊", layout="wide")

# ── 비밀번호 잠금 ─────────────────────────────────────────────────────────────
def _check_password():
    """비밀번호 일치 시 True 반환. Streamlit secrets 또는 기본값 사용."""
    try:
        correct_pw = st.secrets.get("PASSWORD", "9400")
    except Exception:
        correct_pw = "9400"   # 로컬 실행 시 기본 비밀번호

    def _submit():
        if st.session_state.get("_pw_input") == correct_pw:
            st.session_state["_pw_ok"] = True
        else:
            st.session_state["_pw_ok"] = False

    if st.session_state.get("_pw_ok"):
        return True

    st.markdown("## 🔒 통계 자동 분석기")
    st.text_input("비밀번호를 입력하세요", type="password",
                  key="_pw_input", on_change=_submit)
    if "_pw_ok" in st.session_state and not st.session_state["_pw_ok"]:
        st.error("비밀번호가 틀렸습니다.")
    st.stop()

_check_password()

st.markdown("""
<style>
  .main-title{font-size:2rem;font-weight:700;color:#2F5496}
  .sub-title{font-size:.95rem;color:#666;margin-bottom:1rem}
  .card{background:#F3F6FB;border-left:4px solid #2F5496;
        padding:8px 14px;border-radius:4px;margin:3px 0}
</style>""", unsafe_allow_html=True)

# ── 모듈 임포트 ───────────────────────────────────────────────────────────────
try:
    from modules.utils       import detect_scale, cronbach_alpha, calc_ave_cr, fmt_p, sig_stars, build_excel
    from modules.nca         import run_nca
    from modules.fsqca       import run_fsqca
    from modules.association import run_association
    from modules.sentiment   import run_sentiment
    from modules.panel       import run_panel
except Exception as e:
    st.error(f"모듈 로드 오류: {e}\n\n앱과 같은 폴더에 modules/ 폴더가 있는지 확인하세요.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 분석 함수들
# ══════════════════════════════════════════════════════════════════════════════

def auto_detect_constructs(df):
    """공통 접두사 기반 구성개념 자동 탐지"""
    likert = [c for c in df.columns if detect_scale(df[c]) == "likert"]
    groups = defaultdict(list)
    for col in likert:
        m = re.match(r'^([A-Za-z]+\d*)(\d)$', col)
        groups[m.group(1) if m else col].append(col)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def _adequate_check(fit):
    """NFI·RFI·IFI·TLI·CFI ≥ .90 AND RMSEA < .049"""
    return (fit.get("NFI",0)>=.90 and fit.get("RFI",0)>=.90 and
            fit.get("IFI",0)>=.90 and fit.get("TLI",0)>=.90 and
            fit.get("CFI",0)>=.90 and fit.get("RMSEA",1)<.049)


def get_composite_scores(df, constructs):
    """구성개념별 문항 평균(합성점수) DataFrame 반환"""
    return pd.DataFrame(
        {lv: df[items].mean(axis=1) for lv, items in constructs.items()})


def calc_mi_approx(model, obs_df, obs_vars):
    """LM-test 근사 기반 수정지수(MI) 계산"""
    try:
        n   = len(obs_df)
        S   = np.cov(obs_df[obs_vars].values.T, ddof=1)
        Sig, _ = model.calc_sigma()
        Sig = Sig + np.eye(len(obs_vars)) * 1e-8
        Si  = np.linalg.inv(Sig)
        G   = Si @ (S - Sig) @ Si
        rows = []
        p = len(obs_vars)
        for i in range(p):
            for j in range(i + 1, p):
                denom = 2 * Si[i, i] * Si[j, j] + 2 * Si[i, j] ** 2
                mi = (n - 1) * G[i, j] ** 2 / max(denom, 1e-8)
                rows.append({"변수1": obs_vars[i], "변수2": obs_vars[j],
                              "MI": round(float(mi), 3)})
        df_mi = pd.DataFrame(rows).sort_values("MI", ascending=False
                                               ).reset_index(drop=True)
        return df_mi
    except Exception:
        return pd.DataFrame()


def run_cfa_with_mi(df, constructs, mi_threshold=3.84, max_mods=200):
    """
    CFA → MI 기반 무한 반복 수정 (R lavaan auto_refit_all_criteria와 동일 로직)
    - 요인분산=1 고정 → 모든 문항 SE·t·p 산출 (참조지표 없음)
    - MI > 3.84 이면 요인 구분 없이 가장 높은 쌍 추가
    - NFI·RFI·IFI·TLI·CFI ≥ .90 AND RMSEA < .049 충족까지 반복
    Returns (result_dict, error_str)
    """
    try:
        from semopy import Model, calc_stats
    except ImportError:
        return None, "semopy 패키지가 필요합니다."

    try:
        item_to_lv = {item: lv for lv, items in constructs.items() for item in items}
        all_items  = [i for items in constructs.values() for i in items]
        data       = df[all_items].dropna()

        # ── 모형 문자열 (요인분산=1 고정 우선, 실패 시 참조지표 방식) ─────────
        meas_lines = [f"  {lv} =~ {' + '.join(items)}"
                      for lv, items in constructs.items()]
        meas_str   = "\n".join(meas_lines)

        def _fit(model_str):
            try:
                m = Model(model_str); m.fit(data); return m
            except Exception:
                return None

        # 요인분산=1 고정 → 참조지표 없이 모든 문항 SE·t·p 산출
        # semopy 버전별로 문법이 다를 수 있으므로 순서대로 시도
        base_str = meas_str   # 기본값(참조지표 방식)
        for _var_syntax in [
            "\n".join(f"  {lv} ~~ 1*{lv}"   for lv in constructs.keys()),
            "\n".join(f"  {lv} ~~ 1 * {lv}" for lv in constructs.keys()),
            "\n".join(f"  {lv} ~~ 1@{lv}"   for lv in constructs.keys()),
        ]:
            _candidate = meas_str + "\n" + _var_syntax
            if _fit(_candidate) is not None:
                base_str = _candidate
                break

        def _extract_fit(m):
            try:
                sd      = calc_stats(m).iloc[0].to_dict()
                chi2    = float(sd.get("chi2", 0))
                dof     = float(sd.get("DoF", 1))
                chi2_bl = float(sd.get("chi2 Baseline", 0))
                dof_bl  = float(sd.get("DoF Baseline", 1))
                pval    = float(sd.get("chi2 p-value", 1))
                cfi     = float(sd.get("CFI", 0))
                tli     = float(sd.get("TLI", 0))
                nfi     = float(sd.get("NFI", 0))
                rmsea   = float(sd.get("RMSEA", 1))
                rfi = ((chi2_bl/dof_bl - chi2/dof) / (chi2_bl/dof_bl)
                       if chi2_bl>0 and dof_bl>0 and dof>0 else 0.0)
                ifi = ((chi2_bl - chi2) / (chi2_bl - dof)
                       if (chi2_bl - dof) > 0 else 0.0)
                return {"χ²": round(chi2,3), "df": int(dof),
                        "χ²/df": round(chi2/dof,3) if dof>0 else "-",
                        "p": round(pval,3),
                        "NFI": round(nfi,3), "RFI": round(rfi,3),
                        "IFI": round(ifi,3), "TLI": round(tli,3),
                        "CFI": round(cfi,3), "RMSEA": round(rmsea,3)}
            except Exception:
                return {}

        # ── 초기 모형 ─────────────────────────────────────────────────────────
        m0 = _fit(base_str)
        if m0 is None:
            return None, "초기 CFA 추정 실패"
        fit0   = _extract_fit(m0)
        mi_df0 = calc_mi_approx(m0, data, all_items)

        # ── MI 기반 반복 수정 (R auto_refit_all_criteria와 동일) ──────────────
        extra_cov   = []
        added_pairs = set()
        mod_log     = []
        m_cur, fit_cur, mi_cur = m0, fit0.copy(), mi_df0.copy()

        for step in range(max_mods):
            if _adequate_check(fit_cur):
                break                           # ✅ 모든 기준 충족

            # 미추가 쌍 중 MI 최대값 선택 (요인 구분 없음 — R과 동일)
            # pandas apply 버그 회피: list comprehension 으로 필터
            high_mi = mi_cur[mi_cur["MI"] > mi_threshold]
            if high_mi.empty:
                break

            already = {tuple(sorted([r["변수1"], r["변수2"]]))
                       for _, r in high_mi.iterrows()
                       if tuple(sorted([r["변수1"], r["변수2"]])) in added_pairs}
            avail = high_mi[
                high_mi.apply(
                    lambda r: tuple(sorted([r["변수1"], r["변수2"]])) not in added_pairs,
                    axis=1)
            ]
            if avail.empty:
                break                           # 더 이상 추가할 MI 없음

            row    = avail.iloc[0]              # MI 내림차순 정렬 → 최대값
            v1, v2 = row["변수1"], row["변수2"]
            pair   = tuple(sorted([v1, v2]))

            new_cov = extra_cov + [(v1, v2)]
            new_str = (base_str + "\n" +
                       "\n".join(f"  {a} ~~ {b}" for a, b in new_cov))
            m_new = _fit(new_str)
            if m_new is None:
                added_pairs.add(pair)           # 실패한 쌍 스킵
                continue
            fit_new = _extract_fit(m_new)

            lv1 = item_to_lv.get(v1, v1)
            lv2 = item_to_lv.get(v2, v2)
            mod_log.append({
                "단계":         step + 1,
                "수정 경로":    f"{v1} ~~ {v2}",
                "소속 요인":    lv1 if lv1 == lv2 else f"{lv1}↔{lv2}",
                "MI":           round(row["MI"], 3),
                "CFI 전→후":    f"{fit_cur.get('CFI','-')} → {fit_new.get('CFI','-')}",
                "RMSEA 전→후":  f"{fit_cur.get('RMSEA','-')} → {fit_new.get('RMSEA','-')}",
                "RFI 전→후":    f"{fit_cur.get('RFI','-')} → {fit_new.get('RFI','-')}",
                "IFI 전→후":    f"{fit_cur.get('IFI','-')} → {fit_new.get('IFI','-')}",
            })
            extra_cov = new_cov
            m_cur, fit_cur = m_new, fit_new
            added_pairs.add(pair)
            mi_cur = calc_mi_approx(m_cur, data, all_items)

        ins     = m_cur.inspect(std_est=True)
        std_col = next((c for c in ins.columns if "std" in c.lower()), ins.columns[-1])
        final_mi = calc_mi_approx(m_cur, data, all_items)  # 수정 후 최종 MI

        return {
            "init_model":   m0,   "mod_model":    m_cur,
            "init_fit":     fit0, "mod_fit":      fit_cur,
            "init_mi":      mi_df0,
            "final_mi":     final_mi,           # 수정 후 최종 MI 테이블
            "mod_log":      pd.DataFrame(mod_log),
            "extra_cov":    extra_cov,
            "was_modified": len(extra_cov) > 0,
            "ins": ins, "std_col": std_col,
            "data": data, "all_items": all_items,
            "base_str": base_str,
        }, None

    except Exception as e:
        return None, f"CFA MI 오류: {e}"


def run_cfa_sem(df, constructs, hypotheses=None):
    """CFA 또는 SEM 실행. 실패 시 (None, None, None, None) 반환"""
    try:
        from semopy import Model, calc_stats
    except ImportError:
        st.error("semopy 패키지가 필요합니다: pip install semopy")
        return None, None, None, None

    try:
        # 모델 문자열 생성
        mm = "\n".join(f"  {lv} =~ {' + '.join(items)}"
                       for lv, items in constructs.items())
        if hypotheses:
            deps = defaultdict(list)
            for s, t in hypotheses:
                deps[t].append(s)
            mm += "\n" + "\n".join(f"  {t} ~ {' + '.join(ss)}"
                                   for t, ss in deps.items())

        all_items = [i for items in constructs.values() for i in items]
        data = df[all_items].dropna()

        model = Model(mm)
        model.fit(data)

        ins = model.inspect(std_est=True)

        # 표준화 계수 컬럼 탐지
        std_col = None
        for col in ins.columns:
            if "std" in col.lower():
                std_col = col
                break
        if std_col is None:
            std_col = ins.columns[-1]  # 마지막 컬럼 사용

        stats = calc_stats(model)

        # semopy 버전별로 키 이름이 다름 → dict 변환 후 유연하게 매칭
        try:
            if hasattr(stats, "iloc"):          # DataFrame
                stats_dict = stats.iloc[0].to_dict()
            elif hasattr(stats, "to_dict"):     # Series
                stats_dict = stats.to_dict()
            else:
                stats_dict = dict(stats)
        except Exception:
            stats_dict = {}

        # χ², df, χ²/df, p, NFI, RFI, IFI, TLI, CFI, RMSEA 계산
        try:
            chi2    = float(stats_dict.get("chi2", 0))
            dof     = float(stats_dict.get("DoF", 1))
            chi2_bl = float(stats_dict.get("chi2 Baseline", 0))
            dof_bl  = float(stats_dict.get("DoF Baseline", 1))
            pval    = float(stats_dict.get("chi2 p-value", 1))
            cfi     = float(stats_dict.get("CFI", 0))
            tli     = float(stats_dict.get("TLI", 0))
            nfi     = float(stats_dict.get("NFI", 0))
            rmsea   = float(stats_dict.get("RMSEA", 1))
            rfi = ((chi2_bl / dof_bl - chi2 / dof) / (chi2_bl / dof_bl)
                   if chi2_bl > 0 and dof_bl > 0 and dof > 0 else 0.0)
            ifi = ((chi2_bl - chi2) / (chi2_bl - dof)
                   if (chi2_bl - dof) > 0 else 0.0)
            fit = {
                "χ²":    round(chi2, 3),
                "df":    int(dof),
                "χ²/df": round(chi2 / dof, 3) if dof > 0 else "-",
                "p":     round(pval, 3),
                "NFI":   round(nfi, 3),
                "RFI":   round(rfi, 3),
                "IFI":   round(ifi, 3),
                "TLI":   round(tli, 3),
                "CFI":   round(cfi, 3),
                "RMSEA": round(rmsea, 3),
            }
        except Exception:
            fit = {}

        return ins, std_col, stats_dict, fit

    except Exception as e:
        st.error(f"CFA/SEM 오류: {e}")
        return None, None, None, None


def _extract_load_df(ins, std_col, constructs):
    """inspect() 결과에서 요인부하량 추출 (semopy 1.x/2.x 공용)
    반환 컬럼: 잠재변수, 측정변수, 표준화계수, SE, t값, p값
    """
    lv_names  = set(constructs.keys())
    all_items = set(i for v in constructs.values() for i in v)

    if (ins["op"] == "=~").any():          # semopy 1.x
        load_df         = ins[ins["op"] == "=~"].copy()
        lv_col, ind_col = "lval", "rval"
    else:                                   # semopy 2.x
        load_df         = ins[ins["lval"].isin(all_items) &
                               ins["rval"].isin(lv_names)].copy()
        lv_col, ind_col = "rval", "lval"

    # 잠재변수 → 측정변수 순서, 비표준화β(Estimate) 제외
    cols_needed = [lv_col, ind_col, std_col, "Std. Err", "z-value", "p-value"]
    cols_needed = [c for c in cols_needed if c in load_df.columns]
    load_df     = load_df[cols_needed].copy()

    rename_map = {lv_col: "잠재변수", ind_col: "측정변수",
                  std_col: "표준화계수",
                  "Std. Err": "SE", "z-value": "t값", "p-value": "p값_raw"}
    load_df = load_df.rename(columns=rename_map)

    # p값 포맷
    if "p값_raw" in load_df.columns:
        load_df["p값"] = load_df["p값_raw"].apply(
            lambda p: fmt_p(p) if str(p).strip() not in ("-", "") else "-")
        load_df = load_df.drop(columns=["p값_raw"])

    # 숫자형 변환 (참조지표는 SE·t값이 None/NaN → "-"로 표시)
    load_df["표준화계수"] = pd.to_numeric(load_df["표준화계수"], errors="coerce").round(3)
    for col in ["SE", "t값"]:
        if col in load_df.columns:
            load_df[col] = load_df[col].apply(
                lambda v: round(float(v), 3)
                if pd.notna(v) and str(v).strip() not in ("None", "", "nan", "-")
                else "-")

    return load_df.reset_index(drop=True), lv_col


def build_cfa_tables(df, constructs, mi_threshold=3.84, max_mods=200):
    """
    CFA 실행 → MI 계산 → 수정모형 반환
    Returns (load_df, rel_df, init_fit, mod_fit, init_mi, final_mi, mod_log, was_modified, extra_cov)
    """
    result, err = run_cfa_with_mi(df, constructs, mi_threshold, max_mods)
    if result is None:
        st.error(err or "CFA 실행 실패")
        return None, None, None, None, None, None, None, False, []

    try:
        ins      = result["ins"]
        std_col  = result["std_col"]
        load_df, lv_col = _extract_load_df(ins, std_col, constructs)

        rel_rows = []
        for lv, items in constructs.items():
            raw = ins[ins[lv_col] == lv][std_col].values
            lam = np.array([v for v in raw
                            if str(v).strip() not in ("-","","nan","None")], dtype=float)
            ave, cr = calc_ave_cr(lam)
            alpha   = cronbach_alpha(df[items])
            rel_rows.append({"잠재변수": lv, "AVE": ave, "CR": cr, "Cronbach α": alpha})

        return (load_df, pd.DataFrame(rel_rows),
                result["init_fit"], result["mod_fit"],
                result["init_mi"],  result.get("final_mi", pd.DataFrame()),
                result["mod_log"],
                result["was_modified"], result["extra_cov"])

    except Exception as e:
        st.error(f"CFA 결과 처리 오류: {e}")
        return None, None, None, None, None, None, None, False, []


def build_sem_table(df, constructs, hypotheses,
                    cfa_extra_cov=None, mi_threshold=3.84, max_mods=200):
    """
    SEM 추정 → 적합도 미충족 시 MI 기반 수정모형 반복
    cfa_extra_cov: CFA 단계에서 확정된 잔차공분산 목록 (2단계 접근법)
    Returns (path_df, init_fit, mod_fit, mod_log, extra_cov, was_modified)
    """
    try:
        from semopy import Model, calc_stats
    except ImportError:
        st.error("semopy 패키지가 필요합니다.")
        return None, None, None, None, [], False

    try:
        item_to_lv = {item: lv for lv, items in constructs.items() for item in items}
        all_items  = [i for items in constructs.values() for i in items]
        lv_names   = set(constructs.keys())
        data       = df[all_items].dropna()

        def _fit_model(model_str):
            try:
                m = Model(model_str); m.fit(data); return m
            except Exception:
                return None

        def _extract_fit(m):
            try:
                sd      = calc_stats(m).iloc[0].to_dict()
                chi2    = float(sd.get("chi2", 0))
                dof     = float(sd.get("DoF", 1))
                chi2_bl = float(sd.get("chi2 Baseline", 0))
                dof_bl  = float(sd.get("DoF Baseline", 1))
                pval    = float(sd.get("chi2 p-value", 1))
                cfi     = float(sd.get("CFI", 0))
                tli     = float(sd.get("TLI", 0))
                nfi     = float(sd.get("NFI", 0))
                rmsea   = float(sd.get("RMSEA", 1))
                rfi = ((chi2_bl/dof_bl - chi2/dof) / (chi2_bl/dof_bl)
                       if chi2_bl>0 and dof_bl>0 and dof>0 else 0.0)
                ifi = ((chi2_bl - chi2) / (chi2_bl - dof)
                       if (chi2_bl - dof) > 0 else 0.0)
                return {"χ²": round(chi2,3), "df": int(dof),
                        "χ²/df": round(chi2/dof,3) if dof>0 else "-",
                        "p": round(pval,3),
                        "NFI": round(nfi,3), "RFI": round(rfi,3),
                        "IFI": round(ifi,3), "TLI": round(tli,3),
                        "CFI": round(cfi,3), "RMSEA": round(rmsea,3)}
            except Exception:
                return {}

        def _extract_paths(m):
            ins     = m.inspect(std_est=True)
            std_col = next((c for c in ins.columns if "std" in c.lower()), ins.columns[-1])
            # semopy 2.x: 구조경로는 lval·rval 모두 잠재변수이면서 op가 측정모형이 아닌 것
            struct  = ins[ins["lval"].isin(lv_names) & ins["rval"].isin(lv_names)].copy()
            cols    = ["lval","rval", std_col, "Std. Err", "z-value", "p-value"]
            cols    = [c for c in cols if c in struct.columns]
            struct  = struct[cols].copy()
            rm      = {"lval":"종속변수","rval":"독립변수",
                       std_col:"표준화β", "Std. Err":"SE",
                       "z-value":"t값", "p-value":"p값_raw"}
            struct  = struct.rename(columns=rm)
            hyp_map = {(s,t): f"H{i+1}" for i,(s,t) in enumerate(hypotheses)}
            struct["가설"] = struct.apply(
                lambda r: hyp_map.get((r["독립변수"],r["종속변수"]),"-"), axis=1)
            struct["경로"] = struct["독립변수"] + " → " + struct["종속변수"]
            def _sp(p):
                try: return float(p) if str(p).strip() not in("-","","nan") else np.nan
                except: return np.nan
            struct["유의성"]  = struct["p값_raw"].apply(lambda p: sig_stars(_sp(p)))
            struct["채택여부"] = struct["p값_raw"].apply(
                lambda p: ("채택" if _sp(p)<0.05 else "기각")
                if not np.isnan(_sp(p)) else "-")
            struct["p값"] = struct["p값_raw"].apply(
                lambda p: fmt_p(_sp(p)) if not np.isnan(_sp(p)) else "-")
            for col in [c for c in ["표준화β","SE","t값"] if c in struct.columns]:
                struct[col] = pd.to_numeric(struct[col], errors="coerce").round(3)
            out = ["가설","경로","표준화β","SE","t값","p값","유의성","채택여부"]
            return struct[[c for c in out if c in struct.columns]].reset_index(drop=True)

        # ── CFA 확정 잔차공분산 + 구조경로로 base_str 구성 (R sem_mi_syntax와 동일) ──
        # cfa_extra_cov: CFA 단계 extra_cov (수정된 잔차공분산 목록)
        cfa_base = (cfa_extra_cov[0] if cfa_extra_cov and isinstance(cfa_extra_cov[0], str)
                    else None)
        # cfa_extra_cov 가 (base_str, extra_pairs) 형태인지 list of tuples인지 처리
        cfa_cov_pairs = cfa_extra_cov if isinstance(cfa_extra_cov, list) and \
                        (not cfa_extra_cov or isinstance(cfa_extra_cov[0], tuple)) \
                        else []

        meas_lines = [f"  {lv} =~ {' + '.join(items)}"
                      for lv, items in constructs.items()]
        meas_str   = "\n".join(meas_lines)
        var_str    = "\n".join(f"  {lv} ~~ 1*{lv}" for lv in constructs.keys())

        cov_part = ("\n" + "\n".join(f"  {a} ~~ {b}" for a,b in cfa_cov_pairs)
                    if cfa_cov_pairs else "")
        deps = defaultdict(list)
        for s, t in hypotheses:
            deps[t].append(s)
        struct_str = "\n".join(f"  {t} ~ {' + '.join(ss)}" for t,ss in deps.items())

        # 요인분산=1 시도, 실패 시 참조지표 방식
        base_with_var = meas_str + "\n" + var_str + cov_part + "\n" + struct_str
        base_no_var   = meas_str + cov_part + "\n" + struct_str
        base_str = base_with_var if _fit_model(base_with_var) is not None else base_no_var

        # ── 초기 SEM ──────────────────────────────────────────────────────────
        m0 = _fit_model(base_str)
        if m0 is None:
            st.error("SEM 초기 추정 실패")
            return None, None, None, None, [], False
        fit0  = _extract_fit(m0)
        path0 = _extract_paths(m0)

        # ── MI 기반 반복 수정 (R auto_refit_sem_criteria와 동일) ──────────────
        extra_cov   = list(cfa_cov_pairs)  # CFA 확정 공분산에서 시작
        added_pairs = {tuple(sorted(p)) for p in cfa_cov_pairs}
        mod_log     = []
        m_cur, fit_cur = m0, fit0.copy()

        for step in range(max_mods):
            if _adequate_check(fit_cur):
                break                           # ✅ 모든 기준 충족

            mi_cur = calc_mi_approx(m_cur, data, all_items)
            avail  = mi_cur[
                (mi_cur["MI"] > mi_threshold) &
                (~mi_cur.apply(
                    lambda r: tuple(sorted([r["변수1"], r["변수2"]])) in added_pairs,
                    axis=1))
            ]
            if avail.empty:
                break

            row    = avail.iloc[0]
            v1, v2 = row["변수1"], row["변수2"]
            pair   = tuple(sorted([v1, v2]))

            new_cov = extra_cov + [(v1, v2)]
            new_cov_str = "\n".join(f"  {a} ~~ {b}" for a,b in new_cov)
            new_str = (meas_str + "\n" + var_str + "\n" + new_cov_str
                       + "\n" + struct_str
                       if "\n" + var_str in base_str
                       else meas_str + "\n" + new_cov_str + "\n" + struct_str)
            m_new = _fit_model(new_str)
            if m_new is None:
                added_pairs.add(pair); continue
            fit_new = _extract_fit(m_new)

            lv1 = item_to_lv.get(v1, v1); lv2 = item_to_lv.get(v2, v2)
            mod_log.append({
                "단계":        step+1,
                "수정 경로":   f"{v1} ~~ {v2}",
                "소속 요인":   lv1 if lv1==lv2 else f"{lv1}↔{lv2}",
                "MI":          round(row["MI"],3),
                "CFI 전→후":   f"{fit_cur.get('CFI','-')}→{fit_new.get('CFI','-')}",
                "RMSEA 전→후": f"{fit_cur.get('RMSEA','-')}→{fit_new.get('RMSEA','-')}",
            })
            extra_cov = new_cov; m_cur = m_new; fit_cur = fit_new
            added_pairs.add(pair)

        path_final = _extract_paths(m_cur)
        was_mod    = len(mod_log) > 0
        init_mi_sem  = calc_mi_approx(m0,   data, all_items)   # 초기 SEM MI
        final_mi_sem = calc_mi_approx(m_cur, data, all_items)  # 최종 SEM MI

        return (path_final, fit0, fit_cur,
                pd.DataFrame(mod_log), extra_cov, was_mod,
                init_mi_sem, final_mi_sem)

    except Exception as e:
        st.error(f"SEM 오류: {e}")
        return None, None, None, None, [], False, pd.DataFrame(), pd.DataFrame()


def run_correlation(df, constructs):
    """잠재변수 상관관계 (복합점수 기반) + √AVE"""
    try:
        comp = {lv: df[items].mean(axis=1) for lv, items in constructs.items()}
        corr = pd.DataFrame(comp).corr().round(3)

        # AVE 계산 (CFA 기반, 실패 시 복합점수 분산 사용)
        sqrt_ave = {}
        try:
            ins, std_col, _, _ = run_cfa_sem(df, constructs)
            if ins is not None:
                lv_names  = set(constructs.keys())
                all_items = set(i for v in constructs.values() for i in v)
                # semopy 2.x: lval=지표, rval=잠재 / 1.x: lval=잠재, op='=~'
                use_rval = not (ins["op"] == "=~").any()
                lv_col = "rval" if use_rval else "lval"
                for lv, items in constructs.items():
                    raw = ins[ins[lv_col] == lv][std_col].values
                    lam = np.array([v for v in raw
                                    if str(v).strip() not in ("-", "", "nan", "None")],
                                   dtype=float)
                    ave, _ = calc_ave_cr(lam)
                    sqrt_ave[lv] = round(float(np.sqrt(ave)), 3)
        except Exception:
            pass

        # 하삼각 + 대각선 √AVE 테이블 구성
        tbl = corr.copy().astype(object)
        for lv in constructs:
            if lv in tbl.columns:
                tbl.loc[lv, lv] = sqrt_ave.get(lv, "-")

        # 상삼각 제거
        for i in range(len(tbl.columns)):
            for j in range(i + 1, len(tbl.columns)):
                tbl.iloc[i, j] = ""

        return tbl.reset_index().rename(columns={"index": "변수"})

    except Exception as e:
        st.error(f"상관관계 분석 오류: {e}")
        return pd.DataFrame()


def run_regression(df, dep, indeps):
    try:
        from scipy import stats as sc
        data = df[[dep] + indeps].dropna()
        if len(data) < len(indeps) + 2:
            return None, "사례 수가 부족합니다."

        X = np.column_stack([np.ones(len(data)), data[indeps].values.astype(float)])
        y = data[dep].values.astype(float)

        b = np.linalg.lstsq(X, y, rcond=None)[0]
        yh  = X @ b
        res = y - yh
        n, k = len(y), len(indeps)
        mse = np.sum(res**2) / (n - k - 1)
        se  = np.sqrt(mse * np.diag(np.linalg.inv(X.T @ X)))
        tv  = b / se
        pv  = 2 * (1 - sc.t.cdf(np.abs(tv), df=n - k - 1))
        r2  = 1 - np.sum(res**2) / np.sum((y - y.mean())**2)
        r2a = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        std_b = b[1:] * (data[indeps].std(ddof=1).values / float(np.std(y, ddof=1)))

        rows = [{"변수":"상수","β":round(b[0],3),"SE":round(se[0],3),
                 "t":round(tv[0],3),"p":fmt_p(pv[0]),"표준화β":"-"}]
        for i, c in enumerate(indeps):
            rows.append({"변수":c,"β":round(b[i+1],3),"SE":round(se[i+1],3),
                         "t":round(tv[i+1],3),"p":fmt_p(pv[i+1]),
                         "표준화β":round(std_b[i],3)})

        return pd.DataFrame(rows), {"R²":round(r2,3),"수정R²":round(r2a,3),"N":n}

    except Exception as e:
        st.error(f"회귀분석 오류: {e}")
        return None, None


def run_ttest(df, gcol, dcol):
    try:
        from scipy import stats as sc
        g = df[gcol].dropna().unique()
        if len(g) != 2:
            return None, f"집단이 {len(g)}개입니다 (2개여야 합니다)."
        a = df[df[gcol] == g[0]][dcol].dropna()
        b = df[df[gcol] == g[1]][dcol].dropna()
        t, p = sc.ttest_ind(a, b)
        return pd.DataFrame([{
            "집단1": f"{g[0]} (n={len(a)}, M={a.mean():.2f})",
            "집단2": f"{g[1]} (n={len(b)}, M={b.mean():.2f})",
            "t값": round(t,3), "p값": fmt_p(p),
            "유의성": sig_stars(p),
            "결론": "집단 간 차이 있음 ✓" if p < 0.05 else "집단 간 차이 없음"
        }]), None
    except Exception as e:
        return None, str(e)


def run_anova(df, gcol, dcol):
    try:
        from scipy import stats as sc
        groups = [grp[dcol].dropna().values
                  for _, grp in df.groupby(gcol)]
        f, p = sc.f_oneway(*groups)
        desc = df.groupby(gcol)[dcol].agg(["count","mean","std"]).round(3)
        desc.columns = ["N","평균","표준편차"]
        result = pd.DataFrame([{
            "F값": round(f,3), "p값": fmt_p(p),
            "유의성": sig_stars(p),
            "결론": "집단 간 차이 있음 ✓" if p < 0.05 else "집단 간 차이 없음"
        }])
        return desc.reset_index(), result
    except Exception as e:
        st.error(f"ANOVA 오류: {e}")
        return None, None


def suggest_analyses(df, constructs):
    n   = len(df.dropna())
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in df.columns if detect_scale(df[c]) in ("binary","categorical")]
    bin_cols = [c for c in df.columns if detect_scale(df[c]) == "binary"]

    txt = []
    for c in df.columns:
        try:
            if df[c].dtype == object:
                avg_len = df[c].dropna().str.len().mean()
                if avg_len and avg_len > 10:
                    txt.append(c)
        except Exception:
            pass

    return {
        "빈도분석":              {"ok": len(cat) > 0,
                                  "reason": f"범주형 변수 {len(cat)}개 감지"},
        "기술통계":              {"ok": len(num) > 0,
                                  "reason": f"수치형 변수 {len(num)}개"},
        "신뢰도 (Cronbach's α)": {"ok": len(constructs) > 0,
                                  "reason": f"구성개념 {len(constructs)}개 탐지"},
        "확인적 요인분석 (CFA)": {"ok": len(constructs) >= 2 and n >= 100,
                                  "reason": f"구성개념 {len(constructs)}개, N={n}"},
        "상관관계 분석":         {"ok": len(constructs) >= 2,
                                  "reason": f"잠재변수 {len(constructs)}개"},
        "구조방정식 (SEM)":      {"ok": len(constructs) >= 3 and n >= 200,
                                  "reason": f"구성개념 {len(constructs)}개, N={n}"},
        "다중회귀분석":          {"ok": len(num) >= 3,
                                  "reason": f"수치형 변수 {len(num)}개"},
        "독립표본 t검정":        {"ok": len(bin_cols) >= 1 and len(num) >= 1,
                                  "reason": f"이분형 집단변수 {len(bin_cols)}개"},
        "일원분산분석 (ANOVA)":  {"ok": len(cat) >= 1 and len(num) >= 1,
                                  "reason": f"범주형 변수 {len(cat)}개"},
        "패널데이터 분석":       {"ok": len(num) >= 3,
                                  "reason": "패널 구조 여부는 연구자가 확인 필요"},
        "NCA":                   {"ok": len(num) >= 2,
                                  "reason": f"수치형 변수 {len(num)}개 (필요조건 분석)"},
        "fsQCA":                 {"ok": len(num) >= 2,
                                  "reason": f"변수 {len(num)}개 (집합 이론적 분석)"},
        "연관분석":              {"ok": len(cat) >= 2,
                                  "reason": f"범주형/이진형 변수 {len(cat)}개"},
        "감정분석":              {"ok": len(txt) > 0,
                                  "reason": f"텍스트 컬럼 {len(txt)}개 감지"},
    }


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="main-title">📊 통계 자동 분석기</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">데이터를 업로드하면 적합한 통계방법을 자동으로 제안합니다.</p>',
            unsafe_allow_html=True)

# ── STEP 1: 파일 업로드 ───────────────────────────────────────────────────────
st.markdown("### 📁 STEP 1. 데이터 업로드")
uploaded = st.file_uploader("Excel(.xlsx/.xls) 또는 CSV 파일", type=["xlsx","xls","csv"])
if not uploaded:
    st.info("파일을 업로드하면 분석이 시작됩니다.")
    st.stop()

try:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"파일 읽기 오류: {e}")
    st.stop()

# 숫자 변환 (숫자로만 이루어진 컬럼만)
for c in df.columns:
    try:
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().sum() == df[c].notna().sum() and df[c].notna().sum() > 0:
            df[c] = converted
    except Exception:
        pass

st.success(f"✅ 파일 로드 완료 — {len(df)}행 × {len(df.columns)}열")
with st.expander("📋 데이터 미리보기 (상위 10행)"):
    st.dataframe(df.head(10), use_container_width=True)

# ── STEP 2: 구성개념 탐지 ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔍 STEP 2. 구성개념 자동 탐지")
constructs_auto = auto_detect_constructs(df)
c1, c2, c3 = st.columns(3)
c1.metric("전체 변수", len(df.columns))
c2.metric("유효 사례", len(df.dropna()))
c3.metric("탐지된 구성개념", len(constructs_auto))

if constructs_auto:
    for lv, items in constructs_auto.items():
        st.markdown(f"- **{lv}** ({len(items)}문항): {', '.join(items)}")
else:
    st.warning("자동 탐지된 구성개념이 없습니다. 아래에서 직접 설정해 주세요.")

with st.expander("✏️ 구성개념 직접 편집 (필요 시)"):
    n_c = st.number_input("구성개념 수", min_value=1, max_value=20,
                           value=max(len(constructs_auto), 1), key="n_constructs")
    auto_k = list(constructs_auto.keys())
    auto_v = list(constructs_auto.values())
    constructs_edit = {}
    for i in range(int(n_c)):
        ca, cb = st.columns([1, 3])
        default_name  = auto_k[i] if i < len(auto_k) else f"LV{i+1}"
        default_items = auto_v[i] if i < len(auto_v) else []
        nm = ca.text_input(f"이름 {i+1}", value=default_name, key=f"cn_{i}")
        it = cb.multiselect(f"문항 {i+1}", df.columns.tolist(),
                             default=default_items, key=f"ci_{i}")
        if nm and len(it) >= 2:
            constructs_edit[nm] = it
    constructs = constructs_edit if constructs_edit else constructs_auto

if not constructs:
    constructs = constructs_auto

# ── STEP 3: 분석 방법 선택 ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### ✅ STEP 3. 분석 방법 선택")
suggestions = suggest_analyses(df, constructs)
selected = {}
for method, info in suggestions.items():
    icon = "✅" if info["ok"] else "⚠️"
    col_a, col_b = st.columns([1, 10])
    selected[method] = col_a.checkbox(f"{icon}", value=info["ok"], key=f"sel_{method}")
    col_b.markdown(
        f'<div class="card"><b>{method}</b>'
        f'&nbsp;<span style="color:#888;font-size:.83rem">{info["reason"]}</span></div>',
        unsafe_allow_html=True)

# ── 분석별 세부 설정 ──────────────────────────────────────────────────────────
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in df.columns if detect_scale(df[c]) in ("binary","categorical")]
lv_list  = list(constructs.keys())

hypotheses = []
if selected.get("구조방정식 (SEM)") and len(lv_list) >= 2:
    st.markdown("---\n#### 🔗 SEM 가설 설정")
    if "hyps" not in st.session_state:
        st.session_state.hyps = [("", "")]
    if st.button("＋ 가설 추가"):
        st.session_state.hyps.append(("", ""))
    to_delete = None
    for i, (sd, td) in enumerate(st.session_state.hyps):
        ca, cb, cc = st.columns([3, 3, 1])
        s = ca.selectbox(f"H{i+1} 독립변수", lv_list,
                         index=lv_list.index(sd) if sd in lv_list else 0, key=f"hs_{i}")
        t = cb.selectbox(f"H{i+1} 종속변수", lv_list,
                         index=lv_list.index(td) if td in lv_list else 0, key=f"ht_{i}")
        if cc.button("삭제", key=f"hd_{i}"):
            to_delete = i
        if s != t:
            hypotheses.append((s, t))
        st.session_state.hyps[i] = (s, t)
    if to_delete is not None and len(st.session_state.hyps) > 1:
        st.session_state.hyps.pop(to_delete)
        st.rerun()

reg_dep, reg_indep = None, []
if selected.get("다중회귀분석"):
    st.markdown("---\n#### 📈 회귀분석 설정")
    reg_dep   = st.selectbox("종속변수", num_cols, key="rd")
    reg_indep = st.multiselect("독립변수", [c for c in num_cols if c != reg_dep], key="ri")

ttest_g, ttest_d = None, None
if selected.get("독립표본 t검정"):
    st.markdown("---\n#### 📊 t검정 설정")
    bin_cols = [c for c in df.columns if detect_scale(df[c]) == "binary"]
    ttest_g = st.selectbox("집단변수 (이분형)", bin_cols or cat_cols or df.columns.tolist(), key="tg")
    ttest_d = st.selectbox("종속변수", num_cols, key="td")

anova_g, anova_d = None, None
if selected.get("일원분산분석 (ANOVA)"):
    st.markdown("---\n#### 📊 ANOVA 설정")
    anova_g = st.selectbox("집단변수", cat_cols or df.columns.tolist(), key="ag")
    anova_d = st.selectbox("종속변수", num_cols, key="ad")

panel_entity, panel_time, panel_dep, panel_indep = None, None, None, []
if selected.get("패널데이터 분석"):
    st.markdown("---\n#### 🗂️ 패널분석 설정")
    ca, cb, cc = st.columns(3)
    panel_entity = ca.selectbox("개체 ID", df.columns.tolist(), key="pe")
    panel_time   = cb.selectbox("시간 변수", df.columns.tolist(), key="pt")
    panel_dep    = cc.selectbox("종속변수", num_cols, key="pd_")
    panel_indep  = st.multiselect("독립변수",
                                   [c for c in num_cols if c not in [panel_dep, panel_entity, panel_time]],
                                   key="pi")

nca_y, nca_x = None, []
nca_df_input = df  # NCA용 데이터프레임 (합성점수 포함 가능)
if selected.get("NCA"):
    st.markdown("---\n#### 🔬 NCA 설정")
    # 구성개념 합성점수 생성 (CFA 이후 사용 권장)
    composite_df = get_composite_scores(df, constructs) if constructs else pd.DataFrame()
    if not composite_df.empty:
        nca_df_input = pd.concat([df, composite_df.add_suffix("_합성")], axis=1)
        st.info("💡 CFA 구성개념 합성점수(문항 평균)가 변수 목록에 추가되었습니다 — 접미사 '_합성'")
    nca_all_num = [c for c in nca_df_input.columns
                   if pd.api.types.is_numeric_dtype(nca_df_input[c])]
    nca_y = st.selectbox("결과변수(Y)", nca_all_num, key="ny")
    nca_x = st.multiselect("조건변수(X)", [c for c in nca_all_num if c != nca_y], key="nx")

fsqca_outcome, fsqca_conds = None, []
if selected.get("fsQCA"):
    st.markdown("---\n#### 🔵 fsQCA 설정")
    fsqca_outcome = st.selectbox("결과변수", num_cols, key="fo")
    fsqca_conds   = st.multiselect("조건변수 (2개 이상)",
                                    [c for c in num_cols if c != fsqca_outcome], key="fc")
    st.caption("보정 기준: 자동 적용 (5%·50%·95% 분위)")

assoc_cols = []
assoc_sup, assoc_conf, assoc_lift = 0.05, 0.3, 1.0
if selected.get("연관분석"):
    st.markdown("---\n#### 🛒 연관분석 설정")
    assoc_cols = st.multiselect("항목 변수 선택",
                                 [c for c in df.columns if detect_scale(df[c]) in ("binary","categorical")],
                                 key="ac")
    a1, a2, a3 = st.columns(3)
    assoc_sup  = a1.slider("최소 지지도",  0.01, 0.5, 0.05, 0.01, key="as_")
    assoc_conf = a2.slider("최소 신뢰도",  0.1,  1.0, 0.3,  0.05, key="ac_")
    assoc_lift = a3.slider("최소 향상도",  1.0,  5.0, 1.0,  0.1,  key="al_")

sent_col, sent_method = None, "dictionary"
if selected.get("감정분석"):
    st.markdown("---\n#### 💬 감정분석 설정")
    txt_cols = [c for c in df.columns if df[c].dtype == object]
    sent_col = st.selectbox("텍스트 컬럼", txt_cols if txt_cols else df.columns.tolist(), key="sc")
    _has_torch = False
    try:
        import transformers; _has_torch = True  # noqa
    except ImportError:
        pass
    if _has_torch:
        sent_method_ui = st.radio("분석 방법",
                                   ["dictionary (KNU 사전, 빠름)", "transformer (딥러닝, 정확)"],
                                   horizontal=True, key="sm")
        sent_method = "dictionary" if "dictionary" in sent_method_ui else "transformer"
    else:
        st.caption("💡 KNU 사전 방식으로 분석합니다 (transformer 미설치)")
        sent_method = "dictionary"

# ── STEP 4: 실행 ──────────────────────────────────────────────────────────────
st.markdown("---")
if not st.button("🚀 분석 실행", type="primary", use_container_width=True):
    st.stop()

xlsx_sheets   = {}
tab_labels    = []
tab_data      = {}
cfa_extra_cov = []   # CFA 확정 잔차공분산 → SEM에 전달

with st.spinner("분석 중입니다..."):

    # 빈도분석
    if selected.get("빈도분석") and cat_cols:
        try:
            freq_res = {}
            for c in cat_cols[:10]:
                f = df[c].value_counts().sort_index()
                pct = (f / f.sum() * 100).round(1)
                fdf = pd.DataFrame({"빈도": f, "백분율(%)": pct, "누적(%)": pct.cumsum().round(1)})
                fdf.index.name = c
                freq_res[c] = fdf
            tab_data["빈도분석"] = freq_res
            tab_labels.append("빈도분석")
            for cn, fdf in freq_res.items():
                out = fdf.reset_index()
                xlsx_sheets[f"빈도_{cn}"[:31]] = (f"빈도분석 — {cn}", out, "")
        except Exception as e:
            st.warning(f"빈도분석 오류: {e}")

    # 기술통계
    if selected.get("기술통계") and num_cols:
        try:
            desc = df[num_cols].describe().T[["count","mean","std","min","max"]].copy()
            desc.columns = ["N","평균","표준편차","최솟값","최댓값"]
            desc["왜도"] = df[num_cols].skew()
            desc["첨도"] = df[num_cols].kurt()
            desc = desc.round(3)
            tab_data["기술통계"] = desc
            tab_labels.append("기술통계")
            xlsx_sheets["기술통계"] = ("기술통계", desc.reset_index().rename(columns={"index":"변수"}), "")
        except Exception as e:
            st.warning(f"기술통계 오류: {e}")

    # 신뢰도
    if selected.get("신뢰도 (Cronbach's α)") and constructs:
        try:
            rel_rows = []
            for lv, items in constructs.items():
                alpha = cronbach_alpha(df[items])
                rel_rows.append({"구성개념": lv, "문항 수": len(items),
                                  "Cronbach α": alpha,
                                  "판정": "충족 ✓" if (not np.isnan(alpha) and alpha >= 0.7) else "미충족 ✗"})
            rel_df = pd.DataFrame(rel_rows)
            tab_data["신뢰도"] = rel_df
            tab_labels.append("신뢰도")
            xlsx_sheets["신뢰도분석"] = ("신뢰도 분석 (Cronbach's α)", rel_df, "※ α ≥ .70 권장")
        except Exception as e:
            st.warning(f"신뢰도 분석 오류: {e}")

    # CFA
    if selected.get("확인적 요인분석 (CFA)") and constructs:
        with st.spinner("CFA 분석 중 (MI 기반 모형수정 포함)..."):
            try:
                loads, rel_df, fit_init, fit_mod, init_mi, final_mi, mod_log, was_mod, extra_cov = \
                    build_cfa_tables(df, constructs)
                if loads is not None:
                    cfa_extra_cov = extra_cov   # SEM 2단계에 전달
                    tab_data["CFA"] = (loads, rel_df, fit_init, fit_mod,
                                       init_mi, final_mi, mod_log, was_mod, extra_cov)
                    tab_labels.append("CFA")
                    xlsx_sheets["표2_CFA"]     = ("CFA — 요인부하량", loads,
                                                    "※ 표준화계수≥.50, AVE≥.50, CR≥.70")
                    xlsx_sheets["표2_신뢰도"] = ("CFA — AVE/CR/α",  rel_df, "")
                    if was_mod and not mod_log.empty:
                        xlsx_sheets["CFA_수정과정"] = ("CFA — MI 기반 수정 과정",
                                                        mod_log, "※ MI>3.84 기준")
                    if not init_mi.empty:
                        xlsx_sheets["CFA_MI초기"] = ("CFA — 초기 수정지수(MI)",
                                                      init_mi.head(20), "")
                    if not final_mi.empty:
                        xlsx_sheets["CFA_MI최종"] = ("CFA — 최종 수정지수(MI)",
                                                      final_mi.head(20), "")
                else:
                    st.warning("⚠️ CFA 결과 생성 실패. semopy 설치 및 구성개념 설정을 확인하세요.")
            except Exception as e:
                st.warning(f"CFA 오류: {e}")

    # 상관관계
    if selected.get("상관관계 분석") and constructs:
        try:
            corr_tbl = run_correlation(df, constructs)
            if not corr_tbl.empty:
                tab_data["상관관계"] = corr_tbl
                tab_labels.append("상관관계")
                xlsx_sheets["표3_상관관계"] = (
                    "상관관계 분석 (대각선=√AVE)", corr_tbl,
                    "※ 대각선: √AVE | 판별타당도 기준: √AVE > 타 변수 상관계수")
        except Exception as e:
            st.warning(f"상관관계 오류: {e}")

    # SEM
    if selected.get("구조방정식 (SEM)") and constructs and hypotheses:
        with st.spinner("SEM 분석 중 (MI 기반 수정 포함)..."):
            try:
                sem_paths, fit_init_sem, fit_mod_sem, mod_log_sem, extra_cov_sem, was_mod_sem, \
                    init_mi_sem, final_mi_sem = \
                    build_sem_table(df, constructs, hypotheses,
                                   cfa_extra_cov=cfa_extra_cov)
                if sem_paths is not None:
                    tab_data["SEM"] = (sem_paths, fit_init_sem, fit_mod_sem,
                                       mod_log_sem, extra_cov_sem, was_mod_sem,
                                       init_mi_sem, final_mi_sem)
                    tab_labels.append("SEM")
                    xlsx_sheets["표4_가설검증"] = (
                        "SEM 가설 검증", sem_paths,
                        "※ ***p<.001 **p<.01 *p<.05 n.s.=유의하지않음", "채택여부")
                    if was_mod_sem and not mod_log_sem.empty:
                        xlsx_sheets["SEM_수정과정"] = (
                            "SEM — MI 기반 수정 과정", mod_log_sem, "※ MI>3.84 기준")
                else:
                    st.warning("⚠️ SEM 결과를 생성하지 못했습니다.")
            except Exception as e:
                st.warning(f"SEM 오류: {e}")

    # 회귀분석
    if selected.get("다중회귀분석") and reg_dep and reg_indep:
        try:
            coef, summ = run_regression(df, reg_dep, reg_indep)
            if coef is not None:
                tab_data["회귀분석"] = (coef, summ)
                tab_labels.append("회귀분석")
                xlsx_sheets["회귀분석"] = (
                    f"다중회귀분석 (종속: {reg_dep})", coef,
                    f"R²={summ['R²']}, 수정R²={summ['수정R²']}, N={summ['N']}")
        except Exception as e:
            st.warning(f"회귀분석 오류: {e}")

    # t검정
    if selected.get("독립표본 t검정") and ttest_g and ttest_d:
        try:
            tr, err = run_ttest(df, ttest_g, ttest_d)
            if err:
                st.warning(f"t검정: {err}")
            elif tr is not None:
                tab_data["t검정"] = tr
                tab_labels.append("t검정")
                xlsx_sheets["t검정"] = (f"독립표본 t검정 ({ttest_g} → {ttest_d})", tr, "")
        except Exception as e:
            st.warning(f"t검정 오류: {e}")

    # ANOVA
    if selected.get("일원분산분석 (ANOVA)") and anova_g and anova_d:
        try:
            adesc, ares = run_anova(df, anova_g, anova_d)
            if adesc is not None:
                tab_data["ANOVA"] = (adesc, ares)
                tab_labels.append("ANOVA")
                xlsx_sheets["ANOVA_기술"] = (f"ANOVA 기술통계 ({anova_g})", adesc, "")
                xlsx_sheets["ANOVA_결과"] = (f"ANOVA 결과",                 ares,  "")
        except Exception as e:
            st.warning(f"ANOVA 오류: {e}")

    # 패널분석
    if selected.get("패널데이터 분석") and panel_dep and panel_indep:
        with st.spinner("패널 분석 중..."):
            try:
                pres, perr = run_panel(df, panel_entity, panel_time, panel_dep, panel_indep)
                if perr:
                    st.warning(perr)
                elif pres:
                    tab_data["패널분석"] = pres
                    tab_labels.append("패널분석")
                    for k, v in pres.items():
                        xlsx_sheets[f"패널_{k}"[:31]] = (f"패널분석 — {k}", v, "")
            except Exception as e:
                st.warning(f"패널분석 오류: {e}")

    # NCA
    if selected.get("NCA") and nca_y and nca_x:
        with st.spinner("NCA 분석 중..."):
            try:
                nca_res, nca_bn = run_nca(nca_df_input, nca_x, nca_y)
                tab_data["NCA"] = (nca_res, nca_bn)
                tab_labels.append("NCA")
                xlsx_sheets["NCA_효과크기"] = (
                    "NCA — 효과크기", nca_res,
                    "※ d≥.1: 작음 | d≥.3: 중간 | d≥.5: 큼 (Dul 2016)")
                if not nca_bn.empty:
                    xlsx_sheets["NCA_병목"] = ("NCA — 병목 테이블", nca_bn, "")
            except Exception as e:
                st.warning(f"NCA 오류: {e}")

    # fsQCA
    if selected.get("fsQCA") and fsqca_outcome and len(fsqca_conds) >= 2:
        with st.spinner("fsQCA 분석 중..."):
            try:
                fsq_res = run_fsqca(df, fsqca_outcome, fsqca_conds, {})
                tab_data["fsQCA"] = fsq_res
                tab_labels.append("fsQCA")
                for k, v in fsq_res.items():
                    if not v.empty:
                        xlsx_sheets[f"fsQCA_{k}"[:31]] = (f"fsQCA — {k}", v, "")
            except Exception as e:
                st.warning(f"fsQCA 오류: {e}")

    # 연관분석
    if selected.get("연관분석") and assoc_cols:
        with st.spinner("연관분석 중..."):
            try:
                fi, ru, aerr = run_association(df, assoc_cols, assoc_sup, assoc_conf, assoc_lift)
                if aerr:
                    st.warning(aerr)
                elif fi is not None:
                    tab_data["연관분석"] = (fi, ru)
                    tab_labels.append("연관분석")
                    xlsx_sheets["연관_빈발항목"] = ("연관분석 — 빈발항목집합", fi, "")
                    if ru is not None and not ru.empty:
                        xlsx_sheets["연관_규칙"] = (
                            "연관분석 — 연관규칙", ru,
                            "※ 향상도 > 1: 양(+)의 연관관계")
            except Exception as e:
                st.warning(f"연관분석 오류: {e}")

    # 감정분석
    if selected.get("감정분석") and sent_col:
        with st.spinner("감정분석 중..."):
            try:
                sent_res, serr = run_sentiment(df, sent_col, sent_method)
                if serr:
                    st.warning(serr)
                elif sent_res:
                    tab_data["감정분석"] = sent_res
                    tab_labels.append("감정분석")
                    xlsx_sheets["감정_요약"] = ("감정분석 — 요약", sent_res["요약"], "")
                    xlsx_sheets["감정_전체"] = (
                        "감정분석 — 전체결과", sent_res["전체결과"].head(500), "")
            except Exception as e:
                st.warning(f"감정분석 오류: {e}")

# ── 결과 출력 ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📋 분석 결과")

if not tab_labels:
    st.warning("실행된 분석이 없습니다. 분석 방법을 선택하고 설정을 완료한 후 다시 시도하세요.")
    st.stop()

FIT_NOTE = "권장 기준: NFI·RFI·IFI·TLI·CFI ≥ .90 | RMSEA < .049 | SRMR ≤ .08"
tabs = st.tabs(tab_labels)

for tab, lbl in zip(tabs, tab_labels):
    with tab:
        c = tab_data[lbl]

        if lbl == "빈도분석":
            for col, fdf in c.items():
                st.markdown(f"**{col}**")
                st.dataframe(fdf, use_container_width=True)

        elif lbl == "기술통계":
            st.dataframe(c, use_container_width=True)

        elif lbl == "신뢰도":
            st.dataframe(c, use_container_width=True)
            st.caption("α ≥ .70: 내적 일관성 확보")

        elif lbl == "CFA":
            loads, rel_df, fit_init, fit_mod, init_mi, final_mi, mod_log, was_mod, extra_cov = c
            mi_df = init_mi   # 하위 호환 유지

            # ── 적합도 기준표 ────────────────────────────────────────────────
            crit_df = pd.DataFrame([
                {"지수": "NFI",   "권장기준": "≥ .90"},
                {"지수": "RFI",   "권장기준": "≥ .90"},
                {"지수": "IFI",   "권장기준": "≥ .90"},
                {"지수": "TLI",   "권장기준": "≥ .90"},
                {"지수": "CFI",   "권장기준": "≥ .90"},
                {"지수": "RMSEA", "권장기준": "< .049"},
                {"지수": "SRMR",  "권장기준": "≤ .08"},
            ])
            with st.expander("📌 적합도 권장 기준"):
                st.dataframe(crit_df, use_container_width=True, hide_index=True)

            # ── 초기 vs 수정 적합도 비교 ─────────────────────────────────────
            st.markdown("**모델 적합도 비교**")
            fit_col_order = ["χ²", "df", "χ²/df", "p",
                             "NFI", "RFI", "IFI", "TLI", "CFI", "RMSEA"]

            def _fit_row(label, fit):
                row = {"모형": label}
                for k in fit_col_order:
                    row[k] = fit.get(k, "-")
                return row

            if was_mod:
                fit_cmp = pd.DataFrame([
                    _fit_row("초기 모형", fit_init),
                    _fit_row("수정 모형", fit_mod),
                ])
            else:
                fit_cmp = pd.DataFrame([_fit_row("초기 모형", fit_init)])

            # 기준 미충족 셀 강조
            def _hl_fit(row):
                colors = [""] * len(row)
                checks = {"NFI":.90,"RFI":.90,"IFI":.90,"TLI":.90,"CFI":.90}
                cols = list(row.index)
                for idx, col in enumerate(cols):
                    v = row[col]
                    try:
                        fv = float(v)
                        if col in checks and fv < checks[col]:
                            colors[idx] = "background-color:#FFE0E0"
                        elif col == "RMSEA" and fv >= 0.049:
                            colors[idx] = "background-color:#FFE0E0"
                        elif col in checks and fv >= checks[col]:
                            colors[idx] = "background-color:#E8F5E9"
                        elif col == "RMSEA" and fv < 0.049:
                            colors[idx] = "background-color:#E8F5E9"
                    except Exception:
                        pass
                return colors

            st.dataframe(fit_cmp.style.apply(_hl_fit, axis=1),
                         use_container_width=True)
            st.caption("🟢 기준 충족 | 🔴 기준 미충족 | RMSEA < .049 권장")

            # ── 수정 내역 ─────────────────────────────────────────────────────
            if was_mod and extra_cov:
                st.success(f"✅ MI 기반 모형수정 {len(extra_cov)}회 적용")
                paths_str = " / ".join(f"{a} ~~ {b}" for a, b in extra_cov)
                st.markdown(f"**수정된 경로:** `{paths_str}`")
                with st.expander("📋 수정 과정 단계별 상세"):
                    st.dataframe(mod_log, use_container_width=True)
                    st.caption("※ 같은 요인 내 문항 간 잔차공분산만 추가 (이론적 허용 범위)\n"
                               "※ MI > 3.84 (χ² 임계값, p < .05) 기준 적용")
            elif not was_mod:
                all_ok = (fit_init.get("NFI",0)>=.90 and fit_init.get("RFI",0)>=.90 and
                          fit_init.get("IFI",0)>=.90 and fit_init.get("TLI",0)>=.90 and
                          fit_init.get("CFI",0)>=.90 and fit_init.get("RMSEA",1)<.049)
                if all_ok:
                    st.success("✅ 초기 모형 적합도 양호 — 수정 불필요")
                else:
                    st.info("ℹ️ 적합도 기준 미충족이나 MI > 3.84 쌍이 없어 더 이상 수정 불가 — 최종 MI 표를 확인하세요")

            # ── MI 표 (초기 / 최종) ───────────────────────────────────────────
            col_mi1, col_mi2 = st.columns(2)
            with col_mi1:
                if init_mi is not None and not init_mi.empty:
                    with st.expander("🔍 초기 수정지수(MI) 상위 20개"):
                        d = init_mi.head(20).copy()
                        d["판정"] = d["MI"].apply(lambda v: "⚠️ 수정 고려" if v > 3.84 else "")
                        st.dataframe(d, use_container_width=True)
            with col_mi2:
                if final_mi is not None and not final_mi.empty:
                    with st.expander("🔍 최종 수정지수(MI) 상위 20개"):
                        d = final_mi.head(20).copy()
                        d["판정"] = d["MI"].apply(lambda v: "⚠️ 추가 수정 가능" if v > 3.84 else "✅")
                        st.dataframe(d, use_container_width=True)
                        if (final_mi["MI"] > 3.84).any():
                            st.warning("최종 모형에도 MI > 3.84 쌍이 남아 있습니다. 이론적 검토 후 추가 수정을 고려하세요.")
                        else:
                            st.success("모든 잔여 MI < 3.84 — 더 이상 수정 불필요")

            st.markdown("---")
            st.markdown("**CFA 결과** (수정모형 기준)")

            # ── 요인부하량 + AVE/CR/α 통합 테이블 ─────────────────────────────
            try:
                merged  = loads.copy()
                merged["AVE"]       = ""
                merged["CR"]        = ""
                merged["Cronbach α"] = ""
                rel_map = rel_df.set_index("잠재변수")
                seen    = set()
                for idx, row in merged.iterrows():
                    lv = row["잠재변수"]
                    if lv not in seen:
                        seen.add(lv)
                        if lv in rel_map.index:
                            merged.at[idx, "AVE"]        = rel_map.loc[lv, "AVE"]
                            merged.at[idx, "CR"]         = rel_map.loc[lv, "CR"]
                            merged.at[idx, "Cronbach α"] = rel_map.loc[lv, "Cronbach α"]

                disp_cols = ["잠재변수", "측정변수", "표준화계수", "SE", "t값",
                             "p값", "AVE", "CR", "Cronbach α"]
                merged = merged[[c for c in disp_cols if c in merged.columns]]
                st.dataframe(merged, use_container_width=True, hide_index=True)
            except Exception as _e:
                st.dataframe(loads, use_container_width=True)
                st.dataframe(rel_df, use_container_width=True)

            st.caption("권장: 표준화계수 ≥ .50 | AVE ≥ .50 | CR ≥ .70 | Cronbach α ≥ .70")

        elif lbl == "상관관계":
            st.dataframe(c, use_container_width=True)
            st.caption("대각선 = √AVE | 하삼각 = 잠재변수 간 상관계수")

        elif lbl == "SEM":
            paths, fit_init, fit_mod, mod_log, extra_cov, was_mod, init_mi_sem, final_mi_sem = c
            fit_col_order = ["χ²","df","χ²/df","p","NFI","RFI","IFI","TLI","CFI","RMSEA"]

            def _hl_sem(row):
                checks = {"NFI":.90,"RFI":.90,"IFI":.90,"TLI":.90,"CFI":.90}
                colors = []
                for col in row.index:
                    try:
                        fv = float(row[col])
                        if col in checks:
                            colors.append("background-color:#E8F5E9" if fv>=checks[col]
                                          else "background-color:#FFE0E0")
                        elif col == "RMSEA":
                            colors.append("background-color:#E8F5E9" if fv<0.049
                                          else "background-color:#FFE0E0")
                        else:
                            colors.append("")
                    except Exception:
                        colors.append("")
                return colors

            # ── 적합도 비교 ──────────────────────────────────────────────────
            st.markdown("**모델 적합도**")
            if was_mod:
                fit_cmp = pd.DataFrame([
                    {"모형":"초기 모형", **{k: fit_init.get(k,"-") for k in fit_col_order}},
                    {"모형":"수정 모형", **{k: fit_mod.get(k,"-")  for k in fit_col_order}},
                ])
            else:
                fit_cmp = pd.DataFrame([
                    {"모형":"초기 모형", **{k: fit_init.get(k,"-") for k in fit_col_order}}
                ])
            st.dataframe(fit_cmp.style.apply(_hl_sem, axis=1), use_container_width=True)
            st.caption(FIT_NOTE)

            # ── 수정 내역 ────────────────────────────────────────────────────
            if was_mod and extra_cov:
                st.success(f"✅ MI 기반 수정 {len(mod_log)}회 적용")
                paths_str = " / ".join(f"{a} ~~ {b}" for a, b in extra_cov
                                       if isinstance(a, str))
                st.markdown(f"**수정된 경로:** `{paths_str}`")
                with st.expander("📋 수정 과정 상세"):
                    st.dataframe(mod_log, use_container_width=True)
                    st.caption("※ MI > 3.84 (p < .05) 기준 적용")
            elif not was_mod:
                if _adequate_check(fit_init):
                    st.success("✅ 초기 모형 적합도 양호 — 수정 불필요")
                else:
                    st.info("ℹ️ 적합도 미충족이나 MI > 3.84 쌍이 없어 더 이상 수정 불가")

            # ── SEM MI 테이블 (초기 / 최종) ──────────────────────────────────
            col_mi1, col_mi2 = st.columns(2)
            with col_mi1:
                if init_mi_sem is not None and not init_mi_sem.empty:
                    with st.expander("🔍 초기 수정지수(MI) 상위 20개"):
                        d = init_mi_sem.head(20).copy()
                        d["판정"] = d["MI"].apply(lambda v: "⚠️ 수정 고려" if v > 3.84 else "")
                        st.dataframe(d, use_container_width=True)
            with col_mi2:
                if final_mi_sem is not None and not final_mi_sem.empty:
                    with st.expander("🔍 최종 수정지수(MI) 상위 20개"):
                        d = final_mi_sem.head(20).copy()
                        d["판정"] = d["MI"].apply(lambda v: "⚠️ 추가 수정 가능" if v > 3.84 else "✅")
                        st.dataframe(d, use_container_width=True)
                        if (final_mi_sem["MI"] > 3.84).any():
                            st.warning("최종 SEM에도 MI > 3.84 쌍이 남아 있습니다.")
                        else:
                            st.success("모든 잔여 MI < 3.84 — 더 이상 수정 불필요")

            # ── 가설 검증 결과 ────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("**가설 검증 결과** (수정모형 기준)")
            def hl(row):
                color = "#E8F5E9" if row.get("채택여부") == "채택" else "#FFF3F3"
                return [f"background-color:{color}"] * len(row)
            st.dataframe(paths.style.apply(hl, axis=1), use_container_width=True)
            n_adopt = (paths["채택여부"] == "채택").sum()
            st.markdown(f"**채택: {n_adopt}개 / 기각: {len(paths)-n_adopt}개 / 전체: {len(paths)}개**")

        elif lbl == "회귀분석":
            coef, summ = c
            st.dataframe(coef, use_container_width=True)
            st.markdown(f"**R² = {summ['R²']}, 수정R² = {summ['수정R²']}, N = {summ['N']}**")

        elif lbl == "t검정":
            st.dataframe(c, use_container_width=True)

        elif lbl == "ANOVA":
            adesc, ares = c
            st.markdown("**집단별 기술통계**")
            st.dataframe(adesc, use_container_width=True)
            st.markdown("**검정 결과**")
            st.dataframe(ares, use_container_width=True)

        elif lbl == "패널분석":
            for k, v in c.items():
                st.markdown(f"**{k}**")
                st.dataframe(v, use_container_width=True)

        elif lbl == "NCA":
            nca_r, nca_b = c
            st.markdown("**NCA 효과크기 (d)**")
            st.dataframe(nca_r, use_container_width=True)
            st.caption("d < .1: 효과 없음 | d ≥ .1: 작음 | d ≥ .3: 중간 | d ≥ .5: 큼")
            if nca_b is not None and not nca_b.empty:
                st.markdown("**병목 테이블** (X% 달성 시 필요한 최소 Y%)")
                st.dataframe(nca_b, use_container_width=True)

        elif lbl == "fsQCA":
            for k, v in c.items():
                st.markdown(f"**{k}**")
                if v is not None and not v.empty:
                    st.dataframe(v, use_container_width=True)

        elif lbl == "연관분석":
            fi, ru = c
            st.markdown("**빈발항목집합**")
            st.dataframe(fi, use_container_width=True)
            st.markdown("**연관규칙 (향상도 내림차순)**")
            if ru is not None and not ru.empty:
                st.dataframe(ru, use_container_width=True)
            else:
                st.info("조건을 충족하는 연관규칙이 없습니다.")

        elif lbl == "감정분석":
            st.markdown("**감정 분포**")
            st.dataframe(c["요약"], use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**긍정 상위 텍스트**")
                st.dataframe(c["긍정상위"], use_container_width=True)
            with col2:
                st.markdown("**부정 상위 텍스트**")
                st.dataframe(c["부정상위"], use_container_width=True)

# ── Excel 다운로드 ─────────────────────────────────────────────────────────────
st.markdown("---")
if xlsx_sheets:
    try:
        xlsx_bytes = build_excel(xlsx_sheets)
        st.download_button(
            label="📥 RESULT.xlsx 다운로드",
            data=xlsx_bytes,
            file_name="RESULT.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary"
        )
    except Exception as e:
        st.error(f"Excel 생성 오류: {e}")
