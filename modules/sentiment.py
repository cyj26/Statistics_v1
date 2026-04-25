# =============================================================================
# 감정분석 모듈 (한국어 + 영어)
# 방법 1: KNU 한국어 감성사전 (내장) — 오프라인
# 방법 2: HuggingFace transformers — 온라인/GPU 권장
# =============================================================================
import pandas as pd
import numpy as np
import re


# ── KNU 한국어 감성사전 (축약 내장 버전) ──────────────────────────────────────
# 원본: https://github.com/park1200656/KnuSentiLex
_KNU_DICT = {
    # 긍정
    "좋다":2,"훌륭하다":2,"뛰어나다":2,"만족":2,"감사":2,"행복":2,"기쁘다":2,
    "즐겁다":2,"편리하다":2,"탁월하다":2,"우수하다":2,"친절":2,"빠르다":1,
    "깨끗하다":1,"합리적":1,"저렴하다":1,"신뢰":2,"안전하다":1,"편하다":1,
    "최고":2,"훌륭":2,"사랑":2,"좋아":2,"웃음":1,"밝다":1,"성공":2,"발전":1,
    "유익":1,"효과적":1,"혁신":1,"창의":1,"희망":2,"긍정":2,"완벽":2,"탁월":2,
    "추천":1,"칭찬":2,"흡족":2,"감동":2,"유용하다":1,"도움":1,"쉽다":1,
    # 부정
    "나쁘다":-2,"불만":-2,"싫다":-2,"실망":-2,"최악":-2,"불편":-1,"느리다":-1,
    "비싸다":-1,"불안":-1,"위험":-1,"문제":-1,"불량":-2,"고장":-2,"오류":-2,
    "짜증":-2,"화나다":-2,"슬프다":-2,"우울":-2,"부족":-1,"어렵다":-1,
    "힘들다":-1,"복잡하다":-1,"불친절":-2,"무시":-2,"거짓":-2,"사기":-2,
    "불신":-2,"두렵다":-1,"피곤":-1,"지루하다":-1,"부정":-2,"낭비":-1,
    "손해":-2,"후회":-2,"걱정":-1,"아쉽다":-1,"부정적":-2,"억울":-2,
}

# 부정어 패턴
_NEG_PATTERNS = re.compile(r"(안|못|없|않|안됨|불|비|무|미)")


def _knu_score(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    tokens = re.findall(r'\w+', text)
    score  = 0.0
    count  = 0
    for i, tok in enumerate(tokens):
        for word, val in _KNU_DICT.items():
            if word in tok:
                # 앞 토큰에 부정어가 있으면 극성 반전
                neg = (i > 0 and _NEG_PATTERNS.search(tokens[i-1]))
                score += -val if neg else val
                count += 1
    return round(score / max(count, 1), 4)


def _label(score: float, pos_thr: float = 0.3, neg_thr: float = -0.3) -> str:
    if score >= pos_thr:   return "긍정"
    if score <= neg_thr:   return "부정"
    return "중립"


# ── 방법 2: HuggingFace (선택적) ─────────────────────────────────────────────
def _hf_sentiment(texts: list, model_name: str = "snunlp/KR-FinBert-SC"):
    try:
        from transformers import pipeline
        pipe = pipeline("text-classification", model=model_name,
                        tokenizer=model_name, truncation=True, max_length=512)
        results = []
        batch = 16
        for i in range(0, len(texts), batch):
            chunk = [str(t)[:512] for t in texts[i:i+batch]]
            out = pipe(chunk)
            results.extend(out)
        return results
    except Exception as e:
        return [{"error": str(e)}] * len(texts)


# ── 메인 함수 ─────────────────────────────────────────────────────────────────
def run_sentiment(df: pd.DataFrame, text_col: str,
                  method: str = "dictionary",   # "dictionary" | "transformer"
                  model_name: str = "snunlp/KR-FinBert-SC",
                  pos_threshold: float = 0.3,
                  neg_threshold: float = -0.3):
    """
    Parameters
    ----------
    text_col  : 텍스트 컬럼명
    method    : 'dictionary' (KNU 사전, 빠름) | 'transformer' (딥러닝, 정확)
    """
    texts = df[text_col].fillna("").tolist()

    if method == "transformer":
        try:
            import transformers  # noqa
        except ImportError:
            return None, "transformers 패키지가 설치되어 있지 않습니다. dictionary 방법을 사용하세요."
        raw = _hf_sentiment(texts, model_name)
        if "error" in raw[0]:
            return None, f"Transformer 오류: {raw[0]['error']}"
        labels = [r.get("label","").lower() for r in raw]
        scores = [r.get("score", 0.0) for r in raw]
        # 레이블 정규화
        def norm_label(l):
            if "pos" in l or "긍정" in l: return "긍정"
            if "neg" in l or "부정" in l: return "부정"
            return "중립"
        df_out = df[[text_col]].copy()
        df_out["감정레이블"] = [norm_label(l) for l in labels]
        df_out["신뢰도"]    = [round(s, 4) for s in scores]
    else:
        scores = [_knu_score(t) for t in texts]
        labels = [_label(s, pos_threshold, neg_threshold) for s in scores]
        df_out = df[[text_col]].copy()
        df_out["감정점수"]  = scores
        df_out["감정레이블"] = labels

    # 요약 통계
    label_counts = df_out["감정레이블"].value_counts()
    total = len(df_out)
    summary = pd.DataFrame({
        "감정":    label_counts.index,
        "빈도":    label_counts.values,
        "비율(%)": (label_counts.values / total * 100).round(1)
    })

    # 상위 긍정/부정 텍스트 샘플
    if "감정점수" in df_out.columns:
        pos_top = df_out.nlargest(5, "감정점수")[[text_col,"감정점수","감정레이블"]]
        neg_top = df_out.nsmallest(5, "감정점수")[[text_col,"감정점수","감정레이블"]]
    else:
        pos_top = df_out[df_out["감정레이블"]=="긍정"].head(5)
        neg_top = df_out[df_out["감정레이블"]=="부정"].head(5)

    return {
        "전체결과": df_out,
        "요약":    summary,
        "긍정상위": pos_top.reset_index(drop=True),
        "부정상위": neg_top.reset_index(drop=True)
    }, None
