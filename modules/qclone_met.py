def build(ctx):
    """MET(대사당량) 가중 활동 = 에너지소모 프록시 (project8 차용). 활동코드->MET 변환 후
    소모량/강도/피크/변동을 시간대별 집계 + 피험자내 z. Q2(피로) 직접 원인 노림. leak-free."""
    np=ctx['np']; pd=ctx['pd']; sid=ctx['sid']; ld=ctx['ld']; ntr=ctx['ntr']; N=len(sid)
    MET={0:1.3,1:8.0,2:3.5,3:1.2,4:3.0,5:1.5,6:6.0,7:3.5,8:10.0}  # 정지/자전거/도보/정지/차량.../걷기/달리기
    daily={}
    def add(s,d,k,v): daily.setdefault((s,d),{}); daily[(s,d)][k]=daily[(s,d)].get(k,0.0)+v
    perhour={}  # (s,d)->{hour:[mets]}
    try:
        ac=ctx['load']('ch2025_mActivity.parquet'); col=[c for c in ac.columns if c not in ('subject_id','timestamp')][0]
        for s,ts,v in zip(ac['subject_id'].values,ac['timestamp'].values,ac[col].values):
            try: t=pd.Timestamp(ts); a=int(v)
            except Exception: continue
            d=t.normalize(); hr=t.hour; m=MET.get(a,1.5)
            add(s,d,'met_sum',m); add(s,d,'rec',1.0)
            add(s,d,'met_high',1.0 if m>=6 else 0.0)  # 고강도(운동) 횟수
            if 6<=hr<18: add(s,d,'met_day',m)
            if 18<=hr<24: add(s,d,'met_eve',m)
            perhour.setdefault((s,d),{}).setdefault(hr,[]).append(m)
    except Exception: pass
    FCOLS=['met_mean','met_sum','met_high','met_day','met_eve','met_peak','met_std','met_active_hrs','met_conc']
    raw=np.full((N,len(FCOLS)),np.nan)
    for i in range(N):
        try: d=pd.Timestamp(ld[i]).normalize()
        except Exception: continue
        f=daily.get((sid[i],d)); ph=perhour.get((sid[i],d))
        if not f: continue
        rec=max(f.get('rec',0),1)
        o={'met_mean':f.get('met_sum',0)/rec,'met_sum':f.get('met_sum',0),'met_high':f.get('met_high',0),
           'met_day':f.get('met_day',0),'met_eve':f.get('met_eve',0)}
        if ph:
            hourmet=np.array([np.mean(v) for v in ph.values()])
            o['met_peak']=float(hourmet.max()); o['met_std']=float(hourmet.std())
            o['met_active_hrs']=float(np.sum(hourmet>3)); o['met_conc']=float(hourmet.max()/(hourmet.sum()+1e-6))
        raw[i]=[o.get(c,np.nan) for c in FCOLS]
    z=np.full_like(raw,np.nan)
    for s in np.unique(sid):
        idx=np.where(sid==s)[0]; sub=raw[idx]
        with np.errstate(invalid='ignore'): mu=np.nanmean(sub,0); sd=np.nanstd(sub,0)
        sd=np.where((sd==0)|~np.isfinite(sd),1,sd); z[idx]=(sub-mu)/sd
    X=np.hstack([raw,z])
    med=np.nanmedian(X[:ntr],0); med=np.where(np.isfinite(med),med,0)
    inds=np.where(~np.isfinite(X)); X[inds]=np.take(med,inds[1]); X[~np.isfinite(X)]=0
    return X.astype(float)
