def build(ctx):
    """Label-free, transductive 30-minute daily-state representation."""
    np, pd = ctx['np'], ctx['pd']
    sid, ld, ntr = ctx['sid'], ctx['ld'], ctx['ntr']
    N, B, C = len(sid), 48, 6
    keys = {(str(sid[i]), pd.Timestamp(ld[i]).normalize()): i for i in range(N)}
    sums = np.zeros((N, B, C), float)
    cnts = np.zeros((N, B, C), float)

    def locate(s, t):
        t = pd.Timestamp(t); i = keys.get((str(s), t.normalize()))
        return i, (t.hour * 60 + t.minute) // 30

    # ambience: entropy, a robust scalar summary of the acoustic scene
    d = ctx['load']('ch2025_mAmbience.parquet')
    for s, t, a in zip(d.subject_id.values, d.timestamp.values, d.m_ambience.values):
        i, b = locate(s, t)
        if i is None: continue
        try:
            v = np.asarray([float(x[1]) for x in a if np.isfinite(float(x[1]))])
            v = v[v > 0]; v = v / v.sum(); x = -(v * np.log(v)).sum()
        except Exception: continue
        sums[i,b,0] += x; cnts[i,b,0] += 1

    # GPS: speed plus within-record path length
    d = ctx['load']('ch2025_mGps.parquet')
    for s, t, a in zip(d.subject_id.values, d.timestamp.values, d.m_gps.values):
        i, b = locate(s, t)
        if i is None: continue
        try:
            sp = [float(x.get('speed', 0) or 0) for x in a]
            la = np.asarray([float(x['latitude']) for x in a]); lo = np.asarray([float(x['longitude']) for x in a])
            x = np.log1p(np.mean(np.maximum(sp,0)) + 100*np.sum(np.hypot(np.diff(la),np.diff(lo))))
        except Exception: continue
        sums[i,b,1] += x; cnts[i,b,1] += 1

    specs = [('ch2025_mActivity.parquet','m_activity',2),
             ('ch2025_mScreenStatus.parquet','m_screen_use',3)]
    for fn, col, k in specs:
        d = ctx['load'](fn)
        for s,t,x in zip(d.subject_id.values,d.timestamp.values,d[col].values):
            i,b=locate(s,t)
            if i is None: continue
            try: x=float(x)
            except Exception: continue
            sums[i,b,k]+=x; cnts[i,b,k]+=1

    # device cardinality (or scan result count when identifiers are unavailable)
    for fn,col,k in [('ch2025_mWifi.parquet','m_wifi',4),('ch2025_mBle.parquet','m_ble',5)]:
        d=ctx['load'](fn)
        for s,t,a in zip(d.subject_id.values,d.timestamp.values,d[col].values):
            i,b=locate(s,t)
            if i is None: continue
            try: x=np.log1p(len(a))
            except Exception: x=0.
            sums[i,b,k]+=x; cnts[i,b,k]+=1

    X=np.divide(sums,cnts,out=np.full_like(sums,np.nan),where=cnts>0)
    # Only training rows determine normalization/imputation; NMF itself is label-free on all 700 days.
    flat=X.reshape(N*B,C); trmask=np.repeat(np.arange(N)<ntr,B)
    med=np.nanmedian(flat[trmask],0); med=np.where(np.isfinite(med),med,0.)
    q1=np.nanpercentile(flat[trmask],25,axis=0); q3=np.nanpercentile(flat[trmask],75,axis=0)
    scale=np.where(np.isfinite(q3-q1)&((q3-q1)>1e-8),q3-q1,1.)
    ii=np.where(~np.isfinite(flat)); flat[ii]=np.take(med,ii[1])
    Z=np.maximum((flat-med)/scale+2.,0.)
    from sklearn.decomposition import NMF
    nmf=NMF(n_components=8,init='random',random_state=2025,max_iter=400,alpha_W=.03,alpha_H=.03,l1_ratio=.1)
    W=nmf.fit_transform(Z).reshape(N,B,8)
    post=W/(W.sum(2,keepdims=True)+1e-12); state=post.argmax(2)
    occ=post.mean(1)
    trans=(state[:,1:]!=state[:,:-1]).sum(1)[:,None].astype(float)
    ent=(-(occ*np.log(occ+1e-12)).sum(1))[:,None]
    active=(state!=state[:,0,None])
    first=np.where(active.any(1),active.argmax(1),B)[:,None]/B
    last=(B-1-np.flip(active,1).argmax(1))[:,None]/B
    eve=state[:,36:48]; eve_run=np.ones((N,1))
    for i in range(N):
        best=run=1
        for j in range(1,12):
            run=run+1 if eve[i,j]==eve[i,j-1] else 1; best=max(best,run)
        eve_run[i,0]=best/12.
    # subject-specific routine: distance from that subject's mean occupancy (label-free).
    routine=np.zeros((N,1))
    for s in np.unique(sid):
        ix=np.where(sid==s)[0]; mu=occ[ix].mean(0); routine[ix,0]=np.abs(occ[ix]-mu).mean(1)
    raw=np.hstack([occ,trans,first,last,eve_run,routine,ent])
    z=np.zeros_like(raw)
    for s in np.unique(sid):
        ix=np.where(sid==s)[0]; mu=raw[ix].mean(0); sd=raw[ix].std(0); sd=np.where(sd>1e-8,sd,1.)
        z[ix]=(raw[ix]-mu)/sd
    return np.nan_to_num(np.hstack([raw,z]),nan=0.,posinf=0.,neginf=0.)
