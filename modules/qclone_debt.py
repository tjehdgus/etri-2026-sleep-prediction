def build(ctx):
    np = ctx['np']; pd = ctx['pd']
    load = ctx['load']; sid = ctx['sid']; ld = ctx['ld']
    N = 700
    sid = np.asarray(sid)
    ld = pd.to_datetime(pd.Series(list(ld))).reset_index(drop=True)
    def safe_load(name):
        try:
            d = load(name); d = d.copy(); d['ts'] = pd.to_datetime(d['timestamp']); return d
        except Exception: return None
    def hr_clean(x):
        try: a = np.asarray(x, dtype=float)
        except Exception: return np.array([])
        a = a[np.isfinite(a)]; return a[(a >= 30) & (a <= 200)]
    def hr_mean(x):
        a = hr_clean(x); return a.mean() if len(a) >= 3 else np.nan
    def hr_rest(x):
        a = hr_clean(x); return np.percentile(a, 10) if len(a) >= 3 else np.nan
    def hr_cnt(x):
        a = hr_clean(x); return float(len(a))
    def grouped(d):
        g = {}
        if d is None: return g
        try:
            for s, gg in d.groupby('subject_id'): g[s] = gg.sort_values('ts')
        except Exception: pass
        return g
    hr = safe_load('ch2025_wHr.parquet'); pedo = safe_load('ch2025_wPedo.parquet'); act = safe_load('ch2025_mActivity.parquet')
    if hr is not None:
        try:
            hr['hr_m'] = hr['heart_rate'].map(hr_mean); hr['hr_lo'] = hr['heart_rate'].map(hr_rest); hr['hr_n'] = hr['heart_rate'].map(hr_cnt)
        except Exception: hr = None
    g_hr = grouped(hr); g_pe = grouped(pedo); g_ac = grouped(act)
    NDAY = 11; daily = np.full((N, NDAY), np.nan)
    for i in range(N):
        s = sid[i]; d0 = ld.iloc[i]
        if pd.isna(d0): continue
        n_st = d0 + pd.Timedelta(hours=21); n_en = d0 + pd.Timedelta(days=1, hours=9)
        m_st = d0 + pd.Timedelta(days=1, hours=6); d_st = d0 + pd.Timedelta(hours=8); d_en = d0 + pd.Timedelta(hours=22)
        gg = g_hr.get(s)
        if gg is not None:
            try:
                nh = gg[(gg['ts'] >= n_st) & (gg['ts'] < n_en)]
                if len(nh):
                    daily[i, 0] = np.nansum(nh['hr_n'].values)
                    lo = nh['hr_lo'].values; lo = lo[np.isfinite(lo)]
                    if len(lo): daily[i, 1] = np.nanmedian(lo)
                    mn = nh['hr_m'].values; mn = mn[np.isfinite(mn)]
                    if len(mn):
                        daily[i, 8] = np.nanmean(mn); thr = np.nanpercentile(mn, 25); daily[i, 2] = float(np.sum(mn <= thr))
                mh = gg[(gg['ts'] >= m_st) & (gg['ts'] < n_en)]
                if len(mh):
                    mm = mh['hr_m'].values; mm = mm[np.isfinite(mm)]
                    if len(mm): daily[i, 3] = np.nanmean(mm)
                dh = gg[(gg['ts'] >= d_st) & (gg['ts'] < d_en)]
                if len(dh):
                    dm = dh['hr_m'].values; dm = dm[np.isfinite(dm)]
                    if len(dm): daily[i, 7] = np.nanmean(dm); daily[i, 10] = np.nanpercentile(dm, 90)
            except Exception: pass
        gg = g_pe.get(s)
        if gg is not None:
            try:
                dp = gg[(gg['ts'] >= d_st) & (gg['ts'] < d_en)]
                if len(dp):
                    sv = pd.to_numeric(dp['step'], errors='coerce').values; sv = sv[np.isfinite(sv)]
                    if len(sv): daily[i, 4] = np.nansum(sv); daily[i, 5] = np.nanmax(sv)
            except Exception: pass
        gg = g_ac.get(s)
        if gg is not None:
            try:
                da = gg[(gg['ts'] >= d_st) & (gg['ts'] < d_en)]
                if len(da):
                    av = pd.to_numeric(da['m_activity'], errors='coerce').values; av = av[np.isfinite(av)]
                    if len(av): daily[i, 6] = float(np.mean((av != 0) & (av != 3)))
            except Exception: pass
    with np.errstate(all='ignore'):
        cov = daily[:, 0]; rest = daily[:, 1]
        daily[:, 9] = np.where(np.isfinite(cov) & np.isfinite(rest), cov / (1.0 + np.abs(rest)), np.nan)
    def personal_z(col):
        out = np.full(N, np.nan); v = daily[:, col]
        for s in np.unique(sid):
            m = sid == s; x = v[m]; xf = x[np.isfinite(x)]
            if len(xf) >= 2:
                mu = xf.mean(); sg = xf.std() + 1e-6; out[m] = (x - mu) / sg
        return out
    lag1 = np.full((N, NDAY), np.nan); roll3 = np.full((N, NDAY), np.nan)
    debt = np.full(N, np.nan); overload = np.full(N, np.nan)
    z_step = personal_z(4); z_act = personal_z(6); z_rest = personal_z(1); z_qual = personal_z(9); z_morn = personal_z(3)
    for s in np.unique(sid):
        idx = np.where(sid == s)[0]
        if len(idx) == 0: continue
        order = idx[np.argsort(ld.iloc[idx].values)]
        for col in range(NDAY):
            seq = daily[order, col]
            for k, gi in enumerate(order):
                prev = seq[:k]; prev = prev[np.isfinite(prev)]
                if k >= 1 and np.isfinite(seq[k - 1]): lag1[gi, col] = seq[k - 1]
                if len(prev) >= 1: roll3[gi, col] = np.mean(prev[-3:])
        qseq = z_qual[order]; sseq = z_step[order]
        for k, gi in enumerate(order):
            pq = qseq[:k]; pq = pq[np.isfinite(pq)]
            if len(pq) >= 1: debt[gi] = np.mean(np.clip(-pq[-3:], 0, None))
            ps = sseq[:k]; ps = ps[np.isfinite(ps)]
            today = sseq[k] if np.isfinite(sseq[k]) else 0.0; yest = ps[-1] if len(ps) >= 1 else 0.0
            overload[gi] = np.clip(today, 0, None) + np.clip(yest, 0, None)
    recov_deficit = np.nan_to_num(z_rest, nan=0.0) + np.nan_to_num(z_morn, nan=0.0)
    with np.errstate(all='ignore'): morn_drop = daily[:, 3] - daily[:, 1]
    feats = [daily[:,1],daily[:,3],daily[:,0],daily[:,9],daily[:,4],daily[:,6],daily[:,8],morn_drop,
             z_rest,z_morn,z_step,z_act,z_qual,lag1[:,9],lag1[:,1],lag1[:,4],roll3[:,9],roll3[:,1],roll3[:,4],roll3[:,3],debt,overload,recov_deficit]
    X = np.column_stack([np.asarray(f, dtype=float) for f in feats])
    try:
        med = np.nanmedian(X, axis=0); med = np.where(np.isfinite(med), med, 0.0)
        inds = np.where(~np.isfinite(X)); X[inds] = np.take(med, inds[1])
    except Exception: pass
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
