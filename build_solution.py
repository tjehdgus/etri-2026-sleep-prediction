# -*- coding: utf-8 -*-
"""생성기: 기존 검증된 모듈 코드를 그대로 임베드하여 단일 self-contained solution_full.py 생성.
(재작성이 아니라 정확한 기존 코드를 한 파일로 합침 — 대회 단일파일 제출용)"""
import io

# --- 임베드할 모듈 파일들(피처 모듈: build(ctx) 패턴) ---
MODFILES = {
    "daystate": "modules/daystate.py",
    "qclone_trend": "modules/qclone_trend.py",
    "qclone_cross": "modules/qclone_cross.py",
    "qclone_debt": "modules/qclone_debt.py",
    "qclone_met": "modules/qclone_met.py",
    "presleep3": "modules/presleep3.py",
}
mod_src = {}
for k, f in MODFILES.items():
    with open(f, encoding="utf-8") as fh:
        mod_src[k] = fh.read()

# backbone(solution.py)을 베이스로 가져와 pairwise/feat_bank 추가
with open("solution.py", encoding="utf-8") as fh:
    backbone = fh.read()
# backbone의 제출부 직전까지만 사용 (forward-CV 블렌드 전에 pairwise 잔차 삽입)
cut = backbone.index("# ===================== 7) 제출")
head = backbone[:cut]

out = io.StringIO()
out.write(head)
out.write('''
# ===================== 6.5) feat_bank(M) + 피처모듈 + pairwise Q1/Q2 잔차 =====================
# --- 기존 피처 모듈들을 그대로 임베드 (build(ctx) 패턴) ---
_MOD_SRC = {}
''')
for k, src in mod_src.items():
    out.write(f"_MOD_SRC[{k!r}] = " + repr(src) + "\n")

out.write(r'''
def _run_mod(key, ctx):
    ns = {}; exec(_MOD_SRC[key], ns)
    X = np.asarray(ns["build"](ctx), float)
    return np.nan_to_num(X if X.ndim > 1 else X[:, None])

# --- feat_bank M: 밤 생리 20피처 (overnight_search.build_bank 충실 이식) ---
def build_feat_bank():
    IT = ITEMS
    allsid = np.r_[sid, sid_te]
    allld = pd.concat([pd.to_datetime(train_f["lifelog_date"]), pd.to_datetime(test_f["lifelog_date"])]).reset_index(drop=True)
    n = len(allsid)
    hr = pd.read_parquet(f"{IT}/ch2025_wHr.parquet"); hrt = pd.to_datetime(hr["timestamp"])
    def fm(x):
        a = np.asarray(x, float); a = a[(a >= 30) & (a <= 200)]
        if len(a) < 3: return (np.nan, np.nan, np.nan)
        return (a.mean(), a.std(), np.sqrt(np.mean(np.diff(a) ** 2)))
    mm = hr["heart_rate"].apply(fm)
    HR = pd.DataFrame({"sid": hr["subject_id"].values, "ts": hrt.values, "hr": [m[0] for m in mm], "std": [m[1] for m in mm], "rmssd": [m[2] for m in mm]})
    HRg = {s: g.sort_values("ts") for s, g in HR.groupby("sid")}
    ac = pd.read_parquet(f"{IT}/ch2025_mActivity.parquet"); act = pd.to_datetime(ac["timestamp"])
    AC = pd.DataFrame({"sid": ac["subject_id"].values, "min": act.values.astype("datetime64[m]"), "mov": (~pd.to_numeric(ac["m_activity"], errors="coerce").isin([0, 1])).astype(float).values})
    ACg = {s: g.set_index("min")["mov"].groupby(level=0).max() for s, g in AC.groupby("sid")}
    def loadc(fn, col):
        try:
            d = pd.read_parquet(f"{IT}/{fn}"); t = pd.to_datetime(d["timestamp"])
            c = col if col in d.columns else [x for x in d.columns if x not in ("subject_id", "timestamp")][0]
            return pd.DataFrame({"sid": d["subject_id"].values, "ts": t.values, "v": pd.to_numeric(d[c], errors="coerce").values})
        except Exception: return None
    SCR = loadc("ch2025_mScreenStatus.parquet", "screen"); SCRg = {s: g.sort_values("ts") for s, g in SCR.groupby("sid")} if SCR is not None else {}
    LIG = loadc("ch2025_mLight.parquet", "m_light"); LIGg = {s: g.sort_values("ts") for s, g in LIG.groupby("sid")} if LIG is not None else {}
    USE = loadc("ch2025_mUsageStats.parquet", "total_time"); USEg = {s: g.sort_values("ts") for s, g in USE.groupby("sid")} if USE is not None else {}
    cols = ["nhr_mean","nhr_min","nhr_std","rmssd","hrstd","cov","tst","sol","tib","se","waso","nwake","sleephr","circ_amp","circ_mesor","night_mov","scr_night","lig_night","use_night","bedtime_proxy"]
    M = np.full((n, len(cols)), np.nan)
    for idx in range(n):
        s = allsid[idx]; ls = allld.iloc[idx]; g = HRg.get(s)
        if g is None: continue
        start = ls + pd.Timedelta(hours=21); end = start + pd.Timedelta(hours=14)
        w = g[(g["ts"] >= start) & (g["ts"] < end)].dropna(subset=["hr"]); d = {}
        if len(w) >= 20:
            hv = w["hr"].values
            d["nhr_mean"]=hv.mean(); d["nhr_min"]=np.percentile(hv,5); d["nhr_std"]=hv.std()
            d["rmssd"]=np.nanmean(w["rmssd"].values); d["hrstd"]=np.nanmean(w["std"].values); d["cov"]=len(w)
            grid = pd.date_range(start, end, freq="1min")[:-1]
            hs = w.set_index(w["ts"].values.astype("datetime64[m]"))["hr"].groupby(level=0).mean().reindex(grid)
            rest = np.nanpercentile(hs, 20); spn = np.nanpercentile(hs, 65) - rest + 1e-6
            sl = np.clip((spn - (hs - rest)) / spn, 0, 1).fillna(0).values
            acser = ACg.get(s)
            if acser is not None:
                mv = acser.reindex(grid).fillna(0).values; sl = sl * (1 - 0.6 * mv); d["night_mov"] = np.nanmean(mv)
            sm = np.convolve(sl, np.ones(15)/15, mode="same"); asl = sm > 0.5; d["tst"] = asl.sum()
            if asl.sum() >= 10:
                on = np.where(asl)[0]; o0, o1 = on[0], on[-1]; tib = o1 - o0 + 1
                d["sol"]=o0; d["tib"]=tib; d["se"]=asl[o0:o1+1].sum()/tib; d["waso"]=tib-asl[o0:o1+1].sum()
                d["nwake"]=int(np.sum(np.diff(asl[o0:o1+1].astype(int))==-1)); d["sleephr"]=np.nanmean(hs.values[o0:o1+1][asl[o0:o1+1]]); d["bedtime_proxy"]=o0
        wd = g[(g["ts"] >= ls - pd.Timedelta(hours=3)) & (g["ts"] < ls + pd.Timedelta(hours=24))].dropna(subset=["hr"])
        if len(wd) >= 30:
            hh = pd.to_datetime(wd["ts"]).dt.hour + pd.to_datetime(wd["ts"]).dt.minute/60
            Xc = np.c_[np.ones(len(hh)), np.cos(2*np.pi*hh/24), np.sin(2*np.pi*hh/24)]
            try:
                b = np.linalg.lstsq(Xc, wd["hr"].values, rcond=None)[0]; d["circ_amp"]=np.hypot(b[1],b[2]); d["circ_mesor"]=b[0]
            except Exception: pass
        for store, key in [(SCRg,"scr_night"),(LIGg,"lig_night"),(USEg,"use_night")]:
            gg = store.get(s)
            if gg is not None:
                wn = gg[(gg["ts"]>=start)&(gg["ts"]<end)]["v"].values; wn = wn[~np.isnan(wn)]
                if len(wn): d[key] = np.nanmean(wn)
        for ci, c in enumerate(cols):
            if c in d: M[idx, ci] = d[c]
    return M, cols

print("feat_bank(M) 빌드...", flush=True)
_M, _COLS = build_feat_bank()
_sidall = np.r_[sid, sid_te]
_ldall = pd.concat([pd.to_datetime(train_f["lifelog_date"]), pd.to_datetime(test_f["lifelog_date"])]).reset_index(drop=True)
_rawcache = {}
def _loadp(f):
    if f not in _rawcache: _rawcache[f] = pd.read_parquet(os.path.join(ITEMS, f))
    return _rawcache[f]
_ctx = dict(np=np, pd=pd, sid=_sidall, ld=_ldall, ntr=ntr, M=_M.copy(), COLS=list(_COLS),
            load=_loadp, IT=ITEMS, gp=lambda a,b: np.abs((a-b)/np.timedelta64(1,"D")).astype(float))
print("피처모듈 실행(pairwise X)...", flush=True)
_Xpair = np.hstack([_run_mod("qclone_trend",_ctx), _run_mod("qclone_cross",_ctx), _run_mod("qclone_debt",_ctx),
                    _run_mod("qclone_met",_ctx), _run_mod("presleep3",_ctx), _run_mod("daystate",_ctx)])

# --- pairwise Q1/Q2 순위학습 (pairwise_q.py 충실 이식), DSQ12 base 위 α0.15 잔차 ---
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
def _logit(p): p=np.clip(np.asarray(p,float),*CLIP); return np.log(p/(1-p))
def _sig(x): return 1/(1+np.exp(-np.clip(x,-30,30)))
_po, _pp = prior_o, prior_p   # 이미 계산됨
def _outer_folds(n5=5):
    d=train_f.sort_values(["subject_id","lifelog_date"]).reset_index(); blk=max(n5*2,4); by={}
    for s,g in d.groupby("subject_id",sort=False): by[s]=[c for c in np.array_split(g["index"].to_numpy(),blk) if len(c)]
    F=[]
    for f in range(n5):
        v=[]
        for ch in by.values():
            for h in (f,f+n5):
                if h<len(ch): v.append(ch[h])
        v=np.concatenate(v); F.append((np.setdiff1d(d["index"].values,v),v))
    return F
_outer=_outer_folds(5)
def _pair_fit_predict(train_idx, test_idx, j, seed):
    rng=np.random.RandomState(seed); pairs=[]
    for s in np.unique(sid[train_idx]):
        ix=train_idx[sid[train_idx]==s]; pos=ix[y[ix,j]==1]; neg=ix[y[ix,j]==0]
        if not len(pos) or not len(neg): continue
        allp=np.array(np.meshgrid(pos,neg)).T.reshape(-1,2)
        if len(allp)>400: allp=allp[rng.choice(len(allp),400,replace=False)]
        pairs.extend(allp.tolist())
    if not pairs: return _po[test_idx,j]
    p=np.asarray(pairs); D=_Xpair[p[:,0]]-_Xpair[p[:,1]]; D=np.vstack([D,-D]); yy=np.r_[np.ones(len(p)),np.zeros(len(p))]
    sc=StandardScaler().fit(D); m=LogisticRegression(C=.05,max_iter=1000,solver="liblinear").fit(sc.transform(D),yy)
    score=m.decision_function(sc.transform(_Xpair)); out=np.zeros(len(test_idx))
    glob=score[train_idx]
    for k,i in enumerate(test_idx):
        ref=score[train_idx[sid[train_idx]==_sidall[i]]]; ref=ref if len(ref)>=4 else glob
        rank=(np.sum(ref<score[i])+.5*np.sum(ref==score[i])+1)/(len(ref)+2)
        out[k]=_sig(.7*_logit(_po[i,j] if i<ntr else _pp[i-ntr,j])+.3*_logit(rank))
    return np.clip(out,*CLIP)
# pairwise 잔차를 blend(Q1/Q2)에 적용: source=pairwise full-fit, base anchor=blend 자신
print("pairwise Q1/Q2 학습...", flush=True)
_pair_src = {}
for j in (0,1):
    full=np.arange(ntr); testidx=np.arange(ntr,ntr+nte)
    _pair_src[j]=_pair_fit_predict(full,testidx,j,900+j)

# ===================== 7) 제출 =====================
''')
# 제출부: backbone의 blend 계산 후 pairwise 잔차(Q1/Q2) 적용
out.write('''blend2 = blend.copy()
for j in (0,1):
    # DSQ12: daystate Q1/Q2 라우팅을 residual-alpha로 강제 (best 방식), 그 위에 pairwise
    base = C(_sig(_logit(blend[:, j]) + 0.45*(_logit(ds_p[:, j]) - _logit(blend[:, j]))))
    blend2[:, j] = C(_sig(_logit(base) + 0.15*(_logit(_pair_src[j]) - _logit(base))))
outdf = test_f[["subject_id","sleep_date","lifelog_date"]].copy()
for j,l in enumerate(LABELS): outdf[l] = blend2[:, j]
outdf["sleep_date"] = test["sleep_date"]; outdf["lifelog_date"] = test["lifelog_date"]
outdf.to_csv("submission.csv", index=False)
print(f"submission.csv (pairwise 포함) 저장 ({time.time()-t0:.0f}s)", flush=True)
''')

with open("solution_full.py", "w", encoding="utf-8") as fh:
    fh.write(out.getvalue())
print("solution_full.py 생성 완료 (pairwise 포함)")
