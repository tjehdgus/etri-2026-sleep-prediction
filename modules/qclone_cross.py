def build(ctx):
    np = ctx['np']; pd = ctx['pd']
    M = np.asarray(ctx['M'], dtype=float); COLS = list(ctx['COLS'])
    sid = np.asarray(ctx['sid']); ld = pd.to_datetime(pd.Series(list(ctx['ld'])).values); N = M.shape[0]
    def pz(v):
        v = np.asarray(v, dtype=float); out = np.full(N, np.nan)
        for s in np.unique(sid):
            m = sid == s; x = v[m]; xf = x[np.isfinite(x)]
            if len(xf) >= 2: out[m] = (x - xf.mean()) / (xf.std() + 1e-6)
            elif len(xf) == 1: out[m] = 0.0
        return out
    order_of = {}
    for s in np.unique(sid):
        idx = np.where(sid == s)[0]; order_of[s] = idx[np.argsort(ld[idx].astype('datetime64[ns]').astype('int64'))]
    def past_roll(v, win=3):
        v = np.asarray(v, dtype=float); out = np.full(N, np.nan)
        for s, order in order_of.items():
            seq = v[order]
            for k, gi in enumerate(order):
                prev = seq[:k]; prev = prev[np.isfinite(prev)]
                if len(prev) >= 1: out[gi] = np.mean(prev[-win:])
        return out
    def past_lag1(v):
        v = np.asarray(v, dtype=float); out = np.full(N, np.nan)
        for s, order in order_of.items():
            seq = v[order]
            for k, gi in enumerate(order):
                if k >= 1 and np.isfinite(seq[k - 1]): out[gi] = seq[k - 1]
        return out
    low = [str(c).lower() for c in COLS]
    def find(keys, exclude=()):
        return [i for i,c in enumerate(low) if any(k in c for k in keys) and not any(e in c for e in exclude)]
    tst_c=find(['tst','total_sleep','sleep_time','duration']); se_c=find(['se','effic']); sol_c=find(['sol','onset','latency'])
    waso_c=find(['waso','wake','awake','arous']); hr_c=find(['hr','heart','bpm']); cov_c=find(['cov','count','n_','cnt','valid','sample'])
    captured=set(tst_c+se_c+sol_c+waso_c); sleep_generic=[i for i in find(['sleep','s_']) if i not in captured]
    def col_z(idx_list, sign=+1.0):
        zs=[pz(M[:,i])*sign for i in idx_list]
        return np.nanmean(np.column_stack(zs),axis=1) if zs else None
    parts=[]
    for blk,sgn in [(tst_c,+1.0),(se_c,+1.0),(sleep_generic,+1.0),(cov_c,+1.0),(sol_c,-1.0),(waso_c,-1.0)]:
        z=col_z(blk,sgn)
        if z is not None: parts.append(z)
    sleep_prox = np.nanmean(np.column_stack(parts),axis=1) if parts else np.nanmean(np.column_stack([pz(M[:,i]) for i in range(M.shape[1])]),axis=1)
    sleep_prox = np.nan_to_num(sleep_prox, nan=0.0)
    f_sleep_prox=sleep_prox.copy(); f_sleep_prox_z=np.nan_to_num(pz(sleep_prox),nan=0.0)
    debt=np.full(N,np.nan); debt2=np.full(N,np.nan)
    for s,order in order_of.items():
        seq=sleep_prox[order]
        for k,gi in enumerate(order):
            prev=seq[:k]; prev=prev[np.isfinite(prev)]
            if len(prev)>=1: debt[gi]=np.mean(np.clip(-prev[-3:],0,None)); debt2[gi]=np.mean(np.clip(-prev[-2:],0,None))
    f_debt=np.nan_to_num(debt,nan=0.0); f_debt2=np.nan_to_num(debt2,nan=0.0)
    f_prox_trend=np.nan_to_num(sleep_prox-past_roll(sleep_prox,3),nan=0.0)
    night_hr=np.nanmean(np.column_stack([M[:,i] for i in hr_c]),axis=1) if hr_c else np.full(N,np.nan)
    z_hr=pz(night_hr)
    f_recov=np.nan_to_num(z_hr,nan=0.0); f_recov_roll=np.nan_to_num(past_roll(z_hr,3),nan=0.0); f_recov_lag1=np.nan_to_num(past_lag1(z_hr),nan=0.0)
    feats=[f_sleep_prox,f_sleep_prox_z,f_debt,f_debt2,f_prox_trend,f_recov,f_recov_roll,f_recov_lag1]
    for i in range(M.shape[1]):
        try:
            z=pz(M[:,i]); feats.append(np.nan_to_num(z,nan=0.0)); feats.append(np.nan_to_num(past_lag1(z),nan=0.0)); feats.append(np.nan_to_num(past_roll(z,3),nan=0.0))
        except Exception: feats+=[np.zeros(N),np.zeros(N),np.zeros(N)]
    X=np.column_stack([np.asarray(f,dtype=float) for f in feats])
    return np.nan_to_num(X,nan=0.0,posinf=0.0,neginf=0.0)
