# ETRI 2026 수면예측 — 핵심 교훈 (백지 재시작용)

DACON 공유글 기반 코드는 전부 폐기. 이 파일은 **다시 반복하지 말아야 할 것들**만 압축.

## 과제 (사실관계)
- 입력: 스마트폰·워치 센서 12종(parquet, `subject_id,timestamp,값`). `lifelog_date`(전날 하루) → 그날 밤 수면 7지표 예측.
- 타겟 7개(모두 0/1): Q1 수면질·Q2 피로(1=낮음)·Q3 스트레스(1=낮음) = 설문(개인평균 대비), S1 TST·S2 효율·S3 입면·S4 각성 = 수면센서 가이드라인 충족.
- train 450행 / test 250행, **같은 피험자 10명**(id01~10), 날짜만 분리.
- 평가 = **Average(macro) Log Loss** → 잘 보정된 확률 필요(낮을수록 좋음).

## 데이터 함정
- **wHr.heart_rate는 타임스탬프마다 '배열'**(초단위 묶음). 스칼라로 쓰면 전부 NaN됨 → 배열 평균, **30~220 밖 값 제거**.
- 걸음 하루합=0은 대개 "워치 안 참"(결측). 심박 0은 불가능(결측). → 결측 플래그 + 보간/개인평균.
- 센서 커버리지 87~100%(wHr 최저 87%).

## 검증·블렌드 (가장 비싸게 배운 것) ⭐
- **Subject-hole CV**: 피험자별 시간청크에 early+late 구멍 → test 구조 모사, 전체 OOF 커버.
- **블렌드 가중치를 OOF에 맞추면 가짜 점수**가 나온다. 실측 (로컬,LB):
  - v5 단순격자 0.6119 → LB **0.6081** (정합 O)
  - v7 scipy 9소스 연속최적화 0.5992 → LB **0.6231** (❌ 과적합, LB 폭망)
  - v8 **고정가중치** 블렌드 0.6112 → LB **0.61055** (✅ 거의 일치, 차 0.0007)
- **결론: 블렌드는 고정 가중치. 그러면 OOF CV ≈ LB.** 개선은 '피처/모델'로만.
- 워크플로(사용자 원칙): **피처 엔지니어링 완성 → 그다음 모델 선정.** 고정 평가자로 피처 판단.

## 모델
- 개별 단일 최강: **RandomForest / ExtraTrees(얕게, max_depth~6, min_leaf~8)** ≈ 0.616 — LGBM/XGB보다 일반화 좋음.
- 피험자별 라벨평균(prior) = 강한 베이스라인 ≈ 0.627. 블렌드에 항상 섞을 안전판.
- DL(CNN/GRU + 피험자임베딩)은 단독 약함(~0.627)이나 블렌드 다양성엔 기여. 단, 고정블렌드 원칙 지킬 것.

## 성능 한계 (현실)
- within-subject 신호 상한 상관 ~0.14로 약함. **S1~S4는 실제 Withings 수면센서 파생인데 그 센서는 입력에 없음 → 구조적 상한.**
- 정직 CV 바닥 ~0.607. 공개 최고 LB 0.5917(23k 피처). **0.56은 정직하게 도달 불가**(가짜로만 가능).
- 현재 확정 최고 = **LB 0.6081** (단순 4소스 격자 블렌드).

## 환경
- conda env **etri_sleep** (Python 3.11). pandas, numpy, pyarrow, scikit-learn, lightgbm, xgboost, catboost, matplotlib, jupyter, **torch 2.11+cu128(GPU)**.
- GPU: **RTX 5070 Ti 16GB**. `torch.cuda.is_available()=True`.
- 노트북 커널: `Python (etri_sleep)`.
- 경로: `data/ch2026_metrics_train.csv`, `data/ch2026_submission_sample.csv`, `data/ch2025_data_items/*.parquet`.

## 다음 방향(합의)
- 고정 모델/블렌드를 락(lock) → 피처 엔지니어링에 집중, 고정평가자 OOF로만 판단(로컬↓=LB↓).
- 하루 제출 3회 제한. 매 제출 (로컬,LB) 기록.

## 🎯 돌파구 (2026-06-10) — 0.6028 → 0.586

상위팀 슬라이드(ICTC 2025: 마네키네코, sch_csm 1등) 분석으로 정체 돌파. **승리 레시피:**
1. **피처선택 = 모델 중요도(ExtraTrees) ⭐** — 상관계수가 아님. 부스팅이 살아남(상관선택 땐 prior보다 나빴음).
2. **early stopping(30)** — LGBM/CatBoost 과적합 방지.
3. **LGBM(0.3) + CatBoost(0.7) soft voting**, 타겟별 독립.
4. **날씨 데이터** — Open-Meteo 대전(36.35,127.38) 일별 기온/강수/풍속/습도. exp/cache/weather_daejeon.parquet.
5. **중요도 top-k=150** (8000개 과잉 → 노이즈였음. 1등은 779개).
6. **per-target 블렌드** (타겟별 최적 소스 평균).

결과(정직 subject-hole CV): 단일 0.5905 / greedy 0.5896 / **per-target 0.5861**.
- 핵심 코드: exp/recipe.py, exp/recipe_variants.py, exp/recipe_rf.py, exp/final_blend.py
- 제출후보: submit_best_single.csv(0.5905,안전) / submit_blend_greedy.csv / submit_blend_pertarget.csv(0.5861)
- 주의: per-target은 약간 낙관. best_single이 LB 가장 신뢰(LB≈0.59 예상).
- 옛 교훈 정정: "0.60이 한계"는 틀림. 중요도선택+조기종료가 빠진 게 원인이었음.

## 🚀 돌파구 2 (2026-06-12) — 0.6007 → 0.5994 (비전 모델 다양성)

**PixleepFlow 영감(센서→이미지→CNN)으로 0.60 벽 돌파.**
- 핵심: 센서를 2D 이미지(채널×시간)로 변환 → 사전학습 이미지모델 파인튜닝 → **블렌드 다양성 소스로 추가**.
- 이미지모델 단독은 약함(0.66~0.69)이나, 트리/DL과 "다른 시각"이라 블렌드에 큰 기여. 특히 노이즈타겟 Q2/Q3 개선(Q3 0.68→0.66).
- **8소스 탐욕 블렌드**(tree+prior+GRU+BiLSTM+TCN+Attn+Pix18+Pix50): OOF 0.5950 → **LB 0.5994** ✅ 확정 최고.
- 비전모델: torchvision resnet18/50, resnext101 (사전학습 IMAGENET, 앞단 freeze, 시간축roll+노이즈 증강, 64x64). SigLIP(frozen)+LGBM도 추가됨.
- ⚠️ **10소스(+RX101+SigLIP) 탐욕 = OOF 0.5936 but LB 0.6016 (과적합!).** 소스 늘릴수록 탐욕블렌드 과적합. → nested/정규화 블렌드 필요.
- 환경: torchvision 0.26+cu128, transformers 설치됨. 비전 OOF는 pixleep_oof.npz, vision2_oof.npz에 저장.
- 핵심코드: expbase.py(공유), pixleep.py, vision2.py, final_combo.py(8소스, 최고), final_combo2.py(10소스, 과적합).

## 🌙 야간 자율세션 (2026-06-12) — 수면 재구성으로 S1~S4 직격 시도
**현재 확정 최고: LB 0.5994 (submission_combo.csv, 8소스 비전다양성 블렌드).**
- 미세조정(소스추가/nested/전처리/고해상도) 다 0.5994 못 넘음. core6=0.6004, 10소스=0.6016, nested=0.6002.
- 전처리(로그변환): tree엔 효과없음(트리는 이상치 면역, 단조변환 불변). 0.6158→0.6167.
- **근본 가설: 0.54팀은 S1~S4(수면 가이드라인=절대임계값 TST/SE/SOL/WASO)를 정확히 재구성한다.** 우리 exp1은 조잡(심박40%분위)했음.
- 타겟별 우리성능: Q1 0.61/Q2 0.69/Q3 0.69(주관적,노이즈) / S1 0.53/S2 0.57/S3 0.55/S4 0.66.
- 라벨구조: Q1~Q3 개인비율 0.15~0.85로 제각각(단순 50%제약 없음). prior가 subject레벨 잡음.
- **야간작업: 제대로 된 actigraphy 수면재구성(분단위 밤 HR+움직임 → 입면/각성 판정 → TST/SE/SOL/WASO → NSF임계값+피처) → S1~S4 개선 → 블렌드 갱신.**
- 소스 OOF저장: basesrc.npz(tree/prior/GRU/BiLSTM/TCN/Attn), pixleep_oof.npz, vision2_oof.npz, sglarge_oof.npz. 블렌드: final_combo.py(8소스 0.5994).

### 야간 진척 (수면재구성 성공)
- **수면재구성 v2** (움직임주력+최장휴식블록, TST중앙414분/SE0.78 현실적) → S타겟 개선. tree+prior 0.6071→0.6063.
- sleeprecon_v2.py가 핵심. sleep_recon_v2.parquet, sleeptree2_oof.npz(=tree+수면피처 OOF) 저장.
- v1(너무엄격 TST256), v3(TST219)는 v2보다 나쁨. **v2가 최선**.
- **블렌드 탐색(nested 정직지표)**: base 8소스 nested 0.6058. **약한소스 교체가 정답**:
  - A: Pix18→SigLIP_L: nested 0.6040, in-sample 0.5937 (submission_swap.csv)
  - **C: 이미지CNN2개(Pix18,Pix50) → SigLIP_L+SleepTree2: nested 0.6019(최고)/in-sample 0.5939 (submission_C.csv)** ⭐
- C 타겟별: SleepTree2가 S1(0.72)·S3·S4 장악, SigLIP_L이 S2 기여. 수면재구성이 S타겟 직격 입증.
- **미제출 최고 후보: submission_C.csv (nested 0.6019 → LB 0.595~0.598 기대, 0.5994 갱신 유력).** 차선 submission_swap.csv.
- 소스 npz: basesrc, pixleep, vision2, sglarge, sleeptree2, morevis(dino/convnext/effnet). 탐색: blend_search.py.

### 야간 최종 (수면+SigLIP 블렌드 확정)
- sleep_v4(피처과다)·v3 나쁨, sleepseq(밤시퀀스BiLSTM) 단독블렌드 0.6033이나 기존DL과 중복 → 8소스선 SleepTree2가 더 보완적.
- **최종 최고후보 = submission_C.csv** [tree,prior,GRU,BiLSTM,TCN,Attn,SigLIP_L,SleepTree2]: nested 0.6019 / in-sample 0.5939. 여러 비교에서 robust 최고.
- 차선 submission_swap.csv [Pix18→SigLIP_L]: nested 0.6040.
- 기존 확정 LB최고 0.5994(submission_combo.csv 8소스). C는 nested·insample 둘다 개선이라 0.5994 갱신 유력(LB ~0.595~0.598 기대).
- **아침 권장: submission_C.csv 제출.** 결과로 nested↔LB 관계 확정.

## [전력대회 1등 아이디어 이식 — 부검]
- 1등(주머니쥐): TabPFN + 건물별 후진제거 피처선택 + seed앙상블100.
- **피험자별 개별 TabPFN**(엔티티별모델 직역): 단독 nested 0.81(K=30/K=6 동일), 블렌드기여 0. → 피험자당 ~40행(1등은 건물당 ~2000행, 50배), 분리시 교차정보 상실로 부적합. 실증.
- **타겟별 후진제거+TabPFN**(1등 핵심): 같은 folds로 선택시 nested 0.5917 / tree+prior+btarget 0.5831 (착시!). **누수없는 nested-selection 검증: 0.6590**, 블렌드기여 −0.0004(노이즈).
- 교훈: 피처선택을 nested평가와 같은 CV folds에서 하면 선택누수로 ~0.067 낙관편향. submission_BEST(0.5918→LB0.5970) 함정의 재현. **선택은 반드시 outer val 미접촉(inner CV)으로.**
- 결론: submission_C(LB 0.5960) 여전히 확정 best. 1등 두 핵심 모두 데이터규모/누수로 우리문제 이식 실패.

## [정의서 기반 S=임계 재구성 — 음성]
- 2026 정의서: S1~S4 = NSF가이드 단조임계(TST/SE/SOL/WASO), Q=개인 전체기간평균 대비. 정확 임계값은 미공개(데이터서 fit).
- 2025: S1만 3-class(TST 3tier), S2/S3 binary, S4(WASO) 없음. 2025 train=2026 train 동일행(추가데이터 아님, 가치는 S1 세분화뿐).
- 가설: 복원파라미터 1개→isotonic/logistic 단조보정(자연확률). **결과: prior보다도 나쁨**(S1 0.639 vs prior0.581 vs SleepTree2 0.530). 블렌드 0.6019→0.6037 악화.
- 진단: 라벨=1D임계는 맞으나 **병목은 복원정확도**(Withings 실측 없음, proxy센서 추정). 노이즈 파라미터 단독은 기저율도 못이김. 트리(SleepTree2)는 복원을 다피처와 섞어 신호를 건짐 → 이미 천장근처.
- 함의: S지표는 현재 복원 천장 근처. 남은 헤드룸은 Q(주관·피험자내 랭킹).

## [야간 HRV/심박텍스처 — 음성]
- 가설: 초당 심박배열을 평균내 버렸으니 RMSSD/심박하강/각성 추출하면 S복원 향상.
- 실측: expbase가 이미 분내 mean/sd/min/max 4개 추출중. 추가 RMSSD는 sd와 강상관 → 새정보 거의 없음.
- HRV단독 macro 0.76(prior 0.63도 못이김). 블렌드 submission_C 0.6019→+HRV 0.6024(기여 0/미세악화).
- **종합 결론(정직)**: 정의서임계·HR텍스처 등 "큰 베팅" 모두 누수없는 검증서 기여≈0. 프로젝트 내내 동일패턴 — 정직검증시 submission_C(LB0.5960) 못넘음.
- 근본천장: 450행+노이즈 이진7타겟, S=미관측 Withings기기 복원상한, Q=주관(센서예측 한계). 0.596→0.54 "확줄" 경로는 우리 데이터로 사실상 부재. submission_C 확정이 정답.

## [0.56 추격 진단 — 종합]
- 상위60팀 0.56대(일부 소수제출). 우리 LB 0.5960, 격차 0.036(=전체여정 크기).
- 2025 재업 노트북(팀_sch_csm): 피험자단위집계+하드라벨 = prior급(0.62) 베이스라인. 0.56 비법 아님.
- 진단: 풀5000피처 GBM은 시간청크홀드CV서 prior보다 과적합(7중6 악화). 린(소수정예+강정규화+prior)도 0.616에서 막힘. 복잡블렌드 0.596이 여전히 최고. 어떤 honest방법도 0.59~0.62 플래토.
- 센서 전수점검: 숨은 수면데이터 없음(Withings 미제공). S는 반드시 복원.
- 수학: Q2/Q3 주관적=~0.69 고정 → macro 0.56은 S4개를 ~0.40대로 복원해야 가능. 우리 S=0.53~0.65. 즉 0.56팀은 수면복원이 우리보다 압도적이거나 미지의 exploit.
- 랜덤CV가 Q서 낙관(0.667)이나 인접날짜 누수(=우리 함정). 시간청크홀드CV가 LB(0.596) 추적 검증됨 → 신뢰.
- 남은 유일 헤드룸: S복원 정밀화(미사용 신호=활동코드/보행빈도/달리기-걷기/초당HR로 정밀 수면윈도우).

## [정밀 수면복원 v3 — 마지막 베팅, 음성]
- 안쓰던 신호 융합(활동코드 STILL/보행세부/화면/충전/조도/초당HR)로 수면윈도우 정밀화 시도. TST중앙 524분 정상복원.
- 공정비교(둘다 전체트리): v3복원 S평균 0.581 vs SleepTree2(v2) 0.573 → **오히려 미세악화**. 1D calib은 0.63(prior도 못이김).
- 결론: 복원을 어떻게 정밀화해도 S는 0.53~0.58 천장. macro 0.56에 필요한 S~0.40 도달 불가.
- **최종(정직)**: vision/수면복원v2v3super/HRV/TabPFN/타불라NN/외부데이터/per-subject/후진제거/정의서calib/lean — 전부 honest CV서 0.59~0.62 플래토. 0.56 격차는 우리 proxy센서(Withings 미제공)로 도달 불가. submission_C(LB0.5960)가 우리 진짜 천장.

## [돌파구: 라벨 시간/구조 이웃 — 양성!]
- 발견: 라벨 자기상관 존재(Q1+0.27,Q2+0.30,Q3+0.25,S +0.15~0.24). test가 train과 시간교차 → 가까운 train날 라벨이 단서.
- (A)보존exploit은 누수(full평균이 test포함)라 폐기.
- (B)시간이웃: 피험자내 exp(-gap/tau) 가중 라벨평균. 정직CV서 prior 0.627→0.621(Q1/Q2 개선). 센서소스가 못쓰는 직교신호.
- 앙상블 기여(누수없는 nested): 8소스 0.6019 → +Neighbor 0.6003 → +FeatNb 0.5998. **오랜 플래토 첫 돌파.**
- 제출후보: submission_neighbor.csv(9소스, in-sample 0.5913, 예상LB~0.593), submission_neighbor2.csv(10소스 0.5906). submission_C(8소스, LB0.5960)가 baseline.
- nested+insample 둘다 개선 → submission_BEST과 달리 과적합 아님. 구조적 라벨 신호는 더 키울 여지(크로스타겟 상관, 더나은 이웃).
