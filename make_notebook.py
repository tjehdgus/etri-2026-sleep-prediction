# -*- coding: utf-8 -*-
"""solution_full.py -> 단일 self-contained 노트북(solution.ipynb) 변환.
섹션(# =====)별로 코드셀 분할 + 각 섹션 앞에 '어떤 기법인지' 설명 마크다운을 자동 삽입."""
import json, re, io

src = open("solution_full.py", encoding="utf-8").read()

# ============================ 상단 개요 ============================
overview = """# ETRI 2026 휴먼이해 AI — 수면 7지표 예측 (재현 노트북)

스마트폰·웨어러블 센서 라이프로그(전날 하루)로 **그날 밤 수면의 7개 이진 지표**를 예측한다.
- **Q1~Q3**(설문): 수면질/피로/스트레스 — 개인 전체기간 평균 대비 이진화
- **S1~S4**(Withings 수면센서): TST/SE/SOL/WASO의 NSF 가이드라인 준수 여부
- 평가지표: **Average(Macro) Log Loss** → 잘 보정된 확률 출력이 핵심
- train 450 / test 250, 피험자 10명, **train→test 시간적 분포 이동(temporal shift)** 존재

## 실행 방법
- 입력: `./data/` (대회 제공 `ch2026_metrics_train.csv`, `ch2026_submission_sample.csv`, `ch2025_data_items/*.parquet`)
- 출력: `./submission.csv`
- **원본 데이터만으로 처음부터 전체 파이프라인 학습/추론** (외부 산출물 불필요, 약 12분)

## 개발 환경
- OS: Windows-10 (26200) / Python 3.11.15
- numpy 2.4.6 · pandas 2.3.3 · scikit-learn 1.7.2 · lightgbm 4.6.0 · catboost 1.2.10 · torch 2.11.0+cu128 · transformers 4.57.6 · timm 1.0.27

## 사용한 공식 공개 사전학습 모델
- **SigLIP-Large** `google/siglip-large-patch16-256` (HuggingFace, **frozen 임베딩으로만** 사용 — 학습 안 함)
  - 출처: https://huggingface.co/google/siglip-large-patch16-256

## 전체 설계 한눈에
**[다중 소스] → [블렌딩] → [shift-aware 보정]** 의 3단 구조.
1. 서로 다른 관점의 **9개 베이스 소스**(트리/시퀀스DL/비전/수면재구성/이웃)를 만든다.
2. 타겟별 **greedy 블렌딩**으로 합친다.
3. 라벨 정의·시간이동에 맞춘 **잔차 보정**(daystate 라우팅 / pairwise 순위학습 / forward-CV)을 얹는다.

> 핵심 설계 철학: ① 모든 피처를 **피험자내 정규화**(개인 상대성), ② 소규모(N=450)에서 **분산 감소**(DL 다중 seed 평균), ③ test의 **시간이동을 인지한 검증**(forward-chaining CV).
"""

# ============================ 섹션별 '기법 설명' ============================
# solution_full.py 섹션 헤더의 키워드 -> 그 기법이 무엇이고 왜 쓰는지 설명.
SECTION_DOCS = [
    (("센서 로딩", "피처"), """**[기법] 센서 피처 엔지니어링 (피험자내 정규화 + 시계열 파생)**
12개 센서 parquet을 (피험자, lifelog_date) 하루 단위로 집계한다.
- 시간대별 윈도우(all/day/eve/prebed/night/morning/deep) × 통계(mean/std/min/max/median/sum)
- **피험자내 시계열 파생**: lag(diff)·rolling(3)·EMA·expanding mean → "그 사람 평소 대비 오늘"을 인코딩 (Q 라벨이 '개인 평균 대비'라 직결)
- 야간 윈도우는 lifelog_date 기준으로 귀속(취침은 그날 저녁→다음날 새벽)"""),

    (("tree", "prior"), """**[기법] 트리 앙상블 + Pseudo-labeling, 그리고 피험자 사전확률**
- **tree**: LightGBM + CatBoost + ExtraTrees 가중 평균(0.3/0.4/0.3). ExtraTrees 중요도로 상위 150피처 선택, 폴드 OOF.
  - Pseudo-labeling: 1차 예측 중 confident(≥0.85)한 test 행을 학습에 추가(준지도)로 소표본 보강.
- **prior**: 피험자별 라벨 평균(개인 기저율). 단순하지만 강력한 anchor — 특히 S 라벨(NSF 절대기준)은 개인별 충족률이 안정적이라 prior가 강한 베이스라인."""),

    (("4DL", "DLens"), """**[기법] 시퀀스 딥러닝 4종 + DLens(다중 seed 평균 = 분산 감소)**
센서를 24시간×채널 시계열로 만들어 4개 작은 인코더로 학습:
- **GRU / BiLSTM / TCN / Attention** (각각 1D-CNN + 시퀀스 인코더 + 피험자 임베딩 + 멀티태스크 7헤드)
- **DLens**: 같은 모델을 **여러 random seed로 학습해 예측을 평균**한다. 신경망은 고분산 학습기라, N=450 소표본에서 seed 평균이 분산을 줄여 안정적으로 일반화한다(SWA식 아이디어). *이 대회에서 LB로 검증된 몇 안 되는 robust한 이득.*"""),

    (("SigLIP",), """**[기법] 비전 파운데이션 모델(SigLIP-Large) frozen 임베딩**
센서 시계열을 2D 이미지로 만들어 사전학습 **SigLIP-Large** 비전 트랜스포머에 통과 → frozen pooler 임베딩 추출 → LGBM.
- 모델 가중치는 **학습하지 않음(frozen)** — 일반 시각 표현으로 센서 패턴을 다른 각도에서 인코딩(모달 다양성).
- 공식 공개 모델(HuggingFace), 재현 시 자동 다운로드."""),

    (("SleepTree2", "수면재구성"), """**[기법] 수면 에피소드 재구성(SleepTree2)**
야간 움직임(wPedo)·심박(wHr)·충전(mACStatus)으로 **주수면 구간을 재구성** → 분 단위 TST/SE/SOL/WASO 추정 + NSF 임계값 플래그를 피처로 만들어 트리 학습.
- S1~S4(수면지표)를 직접 겨냥한 도메인 피처. 다른 소스와 보완적."""),

    (("daystate", "NMF"), """**[기법] 비지도 일상상태 표현(daystate, transductive NMF)**
train+test 700일을 30분 bin × 6채널로 만들어 **NMF로 일상상태를 분해**(label-free, transductive). 점유율/전이/저녁 루틴/엔트로피 등을 피처화.
- 라벨을 안 쓰므로 test 분포까지 활용. 뒤의 **DSQ12 라우팅**과 **pairwise 피처**의 재료."""),

    (("Neighbor",), """**[기법] 날짜인접 라벨 가중(Neighbor)**
같은 피험자의 **날짜가 가까운 다른 날 라벨**을 지수가중 평균. (OOF는 leave-one-out으로 누수 차단)
- 수면 지표의 날짜 자기상관을 직접 활용하는 단순·강건한 소스."""),

    (("greedy", "블렌드"), """**[기법] 타겟별 greedy 블렌딩 + forward-CV S1/S4 재가중 (★핵심 개선)**
- **greedy coordinate-descent**: 타겟마다 prior에서 출발해 소스 가중치를 0.05/0.1씩 조정하며 logloss 최소화.
- **forward-chaining CV (shift-aware)**: test가 시간적으로 이동(뒤 날짜)했는데, 일반 random-fold CV로 고른 가중치는 일반화가 나쁘다. **앞 날짜로 학습→뒤 날짜 검증**하는 forward 폴드로 **S1/S4 가중치를 다시 선택**하면 held-out 최신 블록에서 일관 개선된다.
  - *본 솔루션의 가장 큰 LB 개선 레버. "random-OOF는 시간이동을 못 본다"를 정면으로 교정.*"""),

    (("feat_bank", "pairwise"), """**[기법] Pairwise Ranking 순위학습 (Q1/Q2) + DSQ12 라우팅**
- **feat_bank**: 밤 생리(심박바닥/HRV/RMSSD/수면추정/취침시각/조도) 20피처 + qclone 파생 모듈들로 Q용 피처 구성.
- **pairwise ranking**: Q 라벨이 '개인 전체평균 대비'라는 **정의에 구조적으로 맞춰**, 같은 피험자의 (양성일, 음성일) 쌍의 **피처 차이(x_pos−x_neg)**로 ridge logistic 순위학습 → 경험적 순위확률 → 잔차(α0.15)로 Q1/Q2에 가산.
- **DSQ12**: daystate를 Q1/Q2에 residual-alpha로 라우팅(개인 일상상태 반영)."""),

    (("제출",), """**[기법] 최종 합성 → 제출 파일**
블렌드 결과에 pairwise 잔차(Q1/Q2)와 DSQ12를 logit 공간에서 합성하고, [0.02, 0.98]로 클리핑하여 `submission.csv` 생성."""),
]

def doc_for(header):
    h = header.replace(" ", "")
    for keys, text in SECTION_DOCS:
        if any(k.replace(" ", "") in h for k in keys):
            return text
    return ""

# ============================ 노트북 조립 ============================
parts = re.split(r"(?m)^(# ={5,}.*$)", src)
cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)})
def code(t):
    t = t.strip("\n")
    if t: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.splitlines(keepends=True)})

def emit_eda():
    """섹션1(피처) 직후 — '왜 이렇게 설계했나'를 원본 데이터만으로 보여주는 동기(EDA) 그림."""
    md("""## 1.5) 설계 동기 (EDA) — 원본 데이터가 말해주는 것

본격적인 모델링 전에, **데이터의 두 가지 사실**이 이후 핵심 설계를 결정한다.
- 그림A(시간이동) → **forward-CV**가 필요한 이유
- 그림B(피험자별 라벨편차) → **피험자내 정규화·pairwise(Q)** vs **prior(S)**를 가른 이유""")
    code('''# 한글 폰트 + 공통 설정 (이후 모든 그림 공통)
import matplotlib, matplotlib.pyplot as plt
from matplotlib import font_manager
_avail={f.name for f in font_manager.fontManager.ttflist}
for _f in ["Malgun Gothic","NanumGothic","Gulim","AppleGothic"]:
    if _f in _avail: matplotlib.rcParams["font.family"]=_f; break
matplotlib.rcParams["axes.unicode_minus"]=False''')

    md("""### 그림A. train→test 시간이동 (← forward-CV의 근거)
test의 sleep_date가 train보다 **뒤쪽 날짜**에 몰려 있다. 무작위 분할 CV는 이 이동을 못 본다 →
그래서 **앞 날짜로 학습→뒤 날짜로 검증**하는 forward-chaining CV로 S1/S4 가중치를 다시 고른다.""")
    code('''fig,ax=plt.subplots(figsize=(9,4))
subs=sorted(set(sid)|set(sid_te)); ymap={s:k for k,s in enumerate(subs)}
ax.scatter(pd.to_datetime(sd),[ymap[s] for s in sid],s=8,alpha=.5,label="train",color="#3bb273")
ax.scatter(pd.to_datetime(sd_te),[ymap[s] for s in sid_te],s=8,alpha=.5,label="test",color="#e1693e")
ax.set_yticks(range(len(subs))); ax.set_yticklabels(subs)
ax.set_xlabel("sleep_date"); ax.set_title("그림A. train→test 시간이동 (test가 뒤 날짜)"); ax.legend()
plt.tight_layout(); plt.show()''')

    md("""### 그림B. 피험자별 라벨 양성률 (← 정규화 vs prior의 근거)
**Q1~Q3**(설문)은 피험자마다 양성률 편차가 크다 → "개인 평균 대비"라는 정의에 맞춰 **피험자내 정규화·pairwise 순위학습**.
**S1~S4**(NSF 절대기준)은 피험자별 충족률이 비교적 안정적 → **prior(개인 기저율)**가 강한 앵커.""")
    code('''_df=pd.DataFrame(y,columns=LABELS); _df["sid"]=sid
rate=_df.groupby("sid")[LABELS].mean()
fig,ax=plt.subplots(figsize=(7,5))
im=ax.imshow(rate.values,aspect="auto",cmap="RdYlBu_r",vmin=0,vmax=1)
ax.set_xticks(range(7)); ax.set_xticklabels(LABELS)
ax.set_yticks(range(len(rate))); ax.set_yticklabels(rate.index)
for a in range(len(rate)):
    for b in range(7): ax.text(b,a,f"{rate.values[a,b]:.2f}",ha="center",va="center",fontsize=7)
ax.set_title("그림B. 피험자별 라벨 양성률 (Q=편차 큼, S=안정)"); plt.colorbar(im,fraction=.04)
plt.tight_layout(); plt.show()''')


def emit_results():
    """파이프라인 끝 — OOF/소스/폴드 변수로 '설계가 실제로 통했는지' 검증하는 결과 그림."""
    md("""## 8) 결과 시각화 — 설계가 통했는지 검증

앞의 동기(EDA)에서 세운 가설이 실제 파이프라인 출력에서 확인되는지 본다.
(끝까지 실행해 만들어진 OOF/소스/폴드 변수를 그대로 사용)
- 그림1 → **다중 소스 블렌드**의 근거(라벨마다 강한 소스가 다름)
- 그림2 → 블렌드가 실제로 무엇에 의존하는지(가중치)
- 그림3 → ★**forward-CV 재가중**(S1/S4)이 가중치를 옮긴 결과
- 그림4 → 소스 **다양성**(낮은 상관 = 블렌드 이득)""")
    code('''SRC=list(src_oof)  # 9개 소스 이름''')

    md("""### 그림1. 소스 × 라벨 OOF logloss (← 다중 소스 블렌드의 근거)
라벨마다 **가장 잘 맞히는 소스가 다르다**(밝을수록 낮은 logloss=좋음). 어떤 단일 소스도 모든 라벨을 지배하지 못함 →
**타겟별 greedy 블렌드**로 라벨마다 최적 조합을 찾는 근거.""")
    code('''ll=np.zeros((len(SRC),7))
for si,s in enumerate(SRC):
    for j in range(7): ll[si,j]=log_loss(y[:,j],C(src_oof[s][:,j]),labels=[0,1])
fig,ax=plt.subplots(figsize=(7,5))
im=ax.imshow(ll,aspect="auto",cmap="viridis_r")
ax.set_xticks(range(7)); ax.set_xticklabels(LABELS)
ax.set_yticks(range(len(SRC))); ax.set_yticklabels(SRC)
for a in range(len(SRC)):
    for b in range(7): ax.text(b,a,f"{ll[a,b]:.3f}",ha="center",va="center",fontsize=6,color="w")
ax.set_title("그림1. 소스×라벨 OOF logloss (낮을수록 좋음)"); plt.colorbar(im,fraction=.04)
plt.tight_layout(); plt.show()''')

    md("""### 그림2. 타겟별 블렌드 가중치
greedy coordinate-descent가 고른 소스 가중치. prior 앵커 위에서 라벨마다 다른 소스를 가져다 쓰는 것을 볼 수 있다.
(Q1/Q2는 daystate 라우팅 포함)""")
    code('''cols=SRC+["daystate"]; W=np.zeros((7,len(cols)))
for j in range(7):
    ol=oof_list9+([ds_o] if j in (0,1) else [])
    w=greedy_weights_for_label(ol,rand_folds,j)
    for i in range(len(SRC)): W[j,i]=w[i]
    if j in (0,1): W[j,-1]=w[-1]
fig,ax=plt.subplots(figsize=(8,5))
im=ax.imshow(W,aspect="auto",cmap="magma")
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols,rotation=45,ha="right")
ax.set_yticks(range(7)); ax.set_yticklabels(LABELS)
for a in range(7):
    for b in range(len(cols)):
        if W[a,b]>0.01: ax.text(b,a,f"{W[a,b]:.2f}",ha="center",va="center",fontsize=6,color="w")
ax.set_title("그림2. 타겟별 블렌드 가중치 (greedy)"); plt.colorbar(im,fraction=.04)
plt.tight_layout(); plt.show()''')

    md("""### 그림3. ★ forward-CV 재가중 (S1/S4) — 동기(그림A)의 검증
같은 소스 집합인데 **무작위 폴드**로 고른 가중치와 **forward 폴드**(앞→뒤)로 고른 가중치가 다르다.
앞의 시간이동(그림A)에 맞춰 가중치를 옮긴 것이 이 솔루션의 **가장 큰 LB 개선 레버**.""")
    code('''fig,axs=plt.subplots(1,2,figsize=(12,4))
for ax,j,nm in zip(axs,[3,6],["S1","S4"]):
    wr=greedy_weights_for_label(oof_list9,rand_folds,j)
    wf=greedy_weights_for_label(oof_list9,forward_folds,j)
    x=np.arange(len(SRC))
    ax.bar(x-.2,wr,.4,label="random-CV",color="#9aa7d0")
    ax.bar(x+.2,wf,.4,label="forward-CV",color="#e1693e")
    ax.set_xticks(x); ax.set_xticklabels(SRC,rotation=45,ha="right")
    ax.set_title(f"{nm}: random vs forward 가중치"); ax.legend()
fig.suptitle("그림3. forward-CV 재가중 효과 (S1/S4)")
plt.tight_layout(); plt.show()''')

    md("""### 그림4. 소스 예측 상관
소스들이 서로 **비상관**일수록 블렌드 이득이 크다. 트리/시퀀스DL/비전(SigLIP)/수면재구성/이웃이
서로 다른 관점이라 상관이 낮게 유지되는 것을 확인.""")
    code('''M=np.column_stack([src_oof[s].reshape(-1) for s in SRC])
corr=np.corrcoef(M.T)
fig,ax=plt.subplots(figsize=(6.5,5.5))
im=ax.imshow(corr,cmap="coolwarm",vmin=-1,vmax=1)
ax.set_xticks(range(len(SRC))); ax.set_xticklabels(SRC,rotation=45,ha="right")
ax.set_yticks(range(len(SRC))); ax.set_yticklabels(SRC)
for a in range(len(SRC)):
    for b in range(len(SRC)): ax.text(b,a,f"{corr[a,b]:.2f}",ha="center",va="center",fontsize=6)
ax.set_title("그림4. 소스 예측 상관 (낮을수록 다양→블렌드 이득)"); plt.colorbar(im,fraction=.04)
plt.tight_layout(); plt.show()''')


md(overview)
code(parts[0])  # imports + 상단 헬퍼
i = 1
while i < len(parts):
    header = parts[i].strip().lstrip("# =").strip()
    body = parts[i + 1] if i + 1 < len(parts) else ""
    doc = doc_for(header)
    md(f"## {header}\n\n{doc}" if doc else f"## {header}")
    code(parts[i] + body)
    if header.startswith("1)"):   # 섹션1(피처) 직후에 설계 동기(EDA) 그림 삽입
        emit_eda()
    i += 2
emit_results()   # 파이프라인 끝에 결과 검증 그림 삽입

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11.15"}},
      "nbformat": 4, "nbformat_minor": 5}
with open("solution.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"solution.ipynb 생성 ({len(cells)} 셀, 기법설명 마크다운 포함)")
