def build(ctx):
    np = ctx['np']; pd = ctx['pd']
    M = ctx['M']; COLS = ctx['COLS']
    sid = np.asarray(ctx['sid'])
    ld = pd.to_datetime(pd.Series(ctx['ld']).values)
    N = M.shape[0]
    def day_agg(fname, how='mean'):
        out = {}
        try: df = ctx['load'](fname)
        except Exception: return out
        if df is None or len(df) == 0: return out
        cols = {c.lower(): c for c in df.columns}
        scol = next((cols[c] for c in ['subject_id','sid','id','user','user_id'] if c in cols), None)
        tcol = next((cols[c] for c in ['timestamp','time','datetime','ts','date','start_time'] if c in cols), None)
        if scol is None or tcol is None: return out
        df = df.copy(); df['_sid'] = df[scol].astype(str); df['_t'] = pd.to_datetime(df[tcol], errors='coerce')
        df = df.dropna(subset=['_t'])
        if len(df) == 0: return out
        df['_date'] = df['_t'].dt.normalize()
        num = df.select_dtypes(include=[np.number]).columns.tolist(); valcols = [c for c in num if c not in ['_sid']]
        if not valcols: return out
        g = df.groupby(['_sid','_date'])[valcols]; agg = g.mean() if how=='mean' else g.sum()
        agg.columns = [f"{fname}_{c}" for c in agg.columns]; return agg
    extra_aggs = []
    for fn, how in [('ch2025_wHr','mean'),('ch2025_wPedo','sum'),('ch2025_mActivity','mean'),('ch2025_mUsageStats','sum'),('ch2025_mScreenStatus','sum'),('ch2025_mLight','mean')]:
        a = day_agg(fn, how=how)
        if isinstance(a, pd.DataFrame) and len(a) > 0: extra_aggs.append(a)
    sid_str = pd.Series(sid).astype(str).values; date_key = pd.Series(ld).dt.normalize().values
    idx = pd.MultiIndex.from_arrays([sid_str, date_key], names=['_sid','_date'])
    extra_signals = {}
    for a in extra_aggs:
        try:
            ar = a.reindex(idx)
            for c in ar.columns: extra_signals[c] = ar[c].values.astype(float)
        except Exception: continue
    base = np.asarray(M, dtype=float); sig_names = list(COLS); sig_mat = base.copy()
    for nm, v in extra_signals.items(): sig_mat = np.column_stack([sig_mat, v]); sig_names.append(nm)
    S = sig_mat.shape[1]
    order_df = pd.DataFrame({'sid': sid_str, 'ld': ld, 'row': np.arange(N)}).sort_values(['sid','ld'], kind='mergesort').reset_index(drop=True)
    sorted_rows = order_df['row'].values; sorted_sid = order_df['sid'].values; sorted_ld = pd.to_datetime(order_df['ld'].values)
    dow = pd.Series(sorted_ld).dt.dayofweek.values; is_weekend = (dow >= 5).astype(float)
    def newcol(): return np.full(N, np.nan, dtype=float)
    group_codes = pd.Series(sorted_sid).factorize()[0]; sorted_sig = sig_mat[sorted_rows]
    def gsr(vals, codes, func, window):
        s = pd.Series(vals); g = s.groupby(codes); sh = g.shift(1); r = sh.groupby(codes)
        if func=='mean': return r.rolling(window,min_periods=1).mean().reset_index(level=0,drop=True).values
        if func=='std': return r.rolling(window,min_periods=2).std().reset_index(level=0,drop=True).values
        if func=='lag1': return sh.values
        if func=='expmean': return r.expanding(min_periods=1).mean().reset_index(level=0,drop=True).values
        if func=='expstd': return r.expanding(min_periods=2).std().reset_index(level=0,drop=True).values
        return np.full(len(vals), np.nan)
    out_cols = []
    def add(name, sv): col = newcol(); col[sorted_rows] = sv; out_cols.append(col)
    eps = 1e-6
    for j in range(S):
        v = sorted_sig[:, j]; nm = sig_names[j]
        ma3=gsr(v,group_codes,'mean',3); ma7=gsr(v,group_codes,'mean',7); sd3=gsr(v,group_codes,'std',3); sd7=gsr(v,group_codes,'std',7)
        lag1=gsr(v,group_codes,'lag1',1); pmean=gsr(v,group_codes,'expmean',1); pstd=gsr(v,group_codes,'expstd',1)
        add(f"{nm}_dev_ma3", v-ma3); add(f"{nm}_dev_ma7", v-ma7); add(f"{nm}_z_personal",(v-pmean)/(pstd+eps))
        add(f"{nm}_rstd3", sd3); add(f"{nm}_rstd7", sd7); add(f"{nm}_chg1", v-lag1); add(f"{nm}_chg1_norm",(v-lag1)/(np.abs(lag1)+eps))
        add(f"{nm}_dev_ma3_over_sd7",(v-ma3)/(sd7+eps)); add(f"{nm}_trend_3v7", ma3-ma7)
    for j in range(S):
        v = sorted_sig[:, j]; s = pd.Series(v); keys = list(zip(group_codes, dow)); kcodes = pd.Series(keys).factorize()[0]
        g = s.groupby(kcodes); dem = g.shift(1).groupby(kcodes).expanding(min_periods=1).mean().reset_index(level=0,drop=True).values
        add(f"{sig_names[j]}_dev_dowmean", v-dem)
    add("is_weekend", is_weekend.astype(float)); add("dow_sin", np.sin(2*np.pi*dow/7.0)); add("dow_cos", np.cos(2*np.pi*dow/7.0))
    day_index = pd.Series(np.ones(N)).groupby(group_codes).cumsum().values - 1; add("person_day_index", day_index)
    X = np.column_stack(out_cols) if out_cols else np.zeros((N,1))
    X = np.where(np.isfinite(X), X, np.nan)
    ntr = ctx.get('ntr', 450); med = np.nanmedian(X[:ntr], axis=0); med = np.where(np.isfinite(med), med, 0.0)
    inds = np.where(~np.isfinite(X)); X[inds] = np.take(med, inds[1]); X = np.clip(X, -1e9, 1e9)
    return X.astype(np.float32)
