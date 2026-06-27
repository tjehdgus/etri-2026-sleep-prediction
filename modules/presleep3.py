def build(ctx):
    """가설A: 개인별 취침시각 추정 -> 취침 직전 윈도우(2h/1h) 고해상도 피처. Q2(피로)/Q3(스트레스) 겨냥.
    취침=저녁 마지막 화면OFF지속+충전시작. 그 직전 폰사용폭증/심박/활동/조도. 피험자내 편차. leak-free."""
    np=ctx['np']; pd=ctx['pd']; sid=ctx['sid']; ld=ctx['ld']; ntr=ctx['ntr']; N=len(sid)
    IT=ctx['IT']
    def H(ts):
        try: return pd.Timestamp(ts)
        except Exception: return None
    # 행 인덱스: (sid, lifelog_date) -- 취침은 그날 저녁
    rowidx={}
    for i in range(N):
        try: rowidx.setdefault((sid[i],pd.Timestamp(ld[i]).normalize()),[]).append(i)
        except Exception: pass
    # 1) 취침시각 추정: 각 (sid,date) 저녁(20시)~새벽(4시) 화면 마지막 ON 시각 + 충전시작
    scr=ctx['load']('ch2025_mScreenStatus.parquet'); scol=[c for c in scr.columns if c not in ('subject_id','timestamp')][0]
    last_on={}  # (sid,date)-> 마지막 화면ON 분(20시기준)
    scr_events={}  # (sid,date)-> [(min_from20, on)]
    for s,ts,v in zip(scr['subject_id'].values,scr['timestamp'].values,scr[scol].values):
        t=H(ts)
        if t is None: continue
        h=t.hour
        # 저녁 d의 20시~ 다음날 11시를 d에 귀속
        if h>=20: d=t.normalize(); mfrom=(h-20)*60+t.minute
        elif h<11: d=(t-pd.Timedelta(days=1)).normalize(); mfrom=(h+4)*60+t.minute
        else: continue
        on=False
        try: on=float(v)>0
        except Exception: pass
        scr_events.setdefault((s,d),[]).append((mfrom,on))
    # 취침시각 = 마지막으로 화면 켜진 뒤 30분+ OFF 지속 시작점 (저녁~새벽)
    bedmin={}
    for k,ev in scr_events.items():
        ev=sorted(ev); on_mins=[m for m,o in ev if o]
        if on_mins: bedmin[k]=max(on_mins)  # 마지막 화면사용 = 취침 근사
    # 2) 취침 직전 윈도우 피처: 폰사용/심박/활동/조도
    def win_feats(loader, fn, valfn, agg='mean'):
        d=loader(fn)
        out={}  # (sid,date)-> {120:[], 60:[]}
        col=None
        for row in d.itertuples(index=False):
            t=H(row.timestamp)
            if t is None: continue
            h=t.hour
            if h>=20: dd=t.normalize(); mfrom=(h-20)*60+t.minute
            elif h<11: dd=(t-pd.Timedelta(days=1)).normalize(); mfrom=(h+4)*60+t.minute
            else: continue
            bm=bedmin.get((row.subject_id,dd))
            if bm is None: continue
            v=valfn(row)
            if v is None: continue
            rel=bm-mfrom  # 취침까지 남은 분 (양수=취침 전)
            if 0<=rel<=120: out.setdefault((row.subject_id,dd),{}).setdefault('w2',[]).append(v)
            if 0<=rel<=60: out.setdefault((row.subject_id,dd),{}).setdefault('w1',[]).append(v)
            if 60<=rel<=150: out.setdefault((row.subject_id,dd),{}).setdefault('wm',[]).append(v)
        return out
    # 폰 앱사용 (취침전)
    def app_val(r):
        try: return sum(float(a.get('total_time',0)) for a in r.m_usage_stats if hasattr(a,'get'))
        except Exception: return None
    app=win_feats(ctx['load'],'ch2025_mUsageStats.parquet',app_val)
    # 화면 on (취침전 폰 만지작)
    def scr_val(r):
        try: return 1.0 if float(getattr(r,scol))>0 else 0.0
        except Exception: return None
    scrw=win_feats(ctx['load'],'ch2025_mScreenStatus.parquet',scr_val)
    # 심박 (취침전)
    def hr_val(r):
        try:
            a=np.asarray(r.heart_rate,float); a=a[np.isfinite(a)]; return float(a.mean()) if len(a) else None
        except Exception: return None
    hrw=win_feats(ctx['load'],'ch2025_wHr.parquet',hr_val)
    # 활동 (취침전 움직임)
    def act_val(r):
        try: a=int(r.m_activity); return 1.0 if a not in (0,3,4) else 0.0
        except Exception: return None
    actw=win_feats(ctx['load'],'ch2025_mActivity.parquet',act_val)
    # 조도 (취침전 빛노출)
    def li_val(r):
        try: return float(np.log1p(max(float(r.m_light),0)))
        except Exception: return None
    liw=win_feats(ctx['load'],'ch2025_mLight.parquet',li_val)
    FCOLS=['bedmin','app_w2','app_w1','scr_w2','scr_w1','hr_w2','hr_w1','act_w2','act_w1','light_w1','app_wm','hr_wm','act_wm','scr_wm']
    raw=np.full((N,len(FCOLS)),np.nan)
    for i in range(N):
        try: dt=pd.Timestamp(ld[i]).normalize()
        except Exception: continue
        s=sid[i]; k=(s,dt)
        f={'bedmin':bedmin.get(k,np.nan)}
        def g(dic,w,how='mean'):
            d=dic.get(k);
            if not d or w not in d: return np.nan
            v=d[w]; return (np.mean(v) if how=='mean' else np.sum(v))
        f['app_w2']=g(app,'w2','sum'); f['app_w1']=g(app,'w1','sum')
        f['scr_w2']=g(scrw,'w2'); f['scr_w1']=g(scrw,'w1')
        f['hr_w2']=g(hrw,'w2'); f['hr_w1']=g(hrw,'w1')
        f['act_w2']=g(actw,'w2'); f['act_w1']=g(actw,'w1')
        f['light_w1']=g(liw,'w1')
        f['app_wm']=g(app,'wm','sum'); f['hr_wm']=g(hrw,'wm'); f['act_wm']=g(actw,'wm'); f['scr_wm']=g(scrw,'wm')
        raw[i]=[f.get(c,np.nan) for c in FCOLS]
    # 피험자내 z
    z=np.full_like(raw,np.nan)
    for s in np.unique(sid):
        ix=np.where(sid==s)[0]; sub=raw[ix]
        with np.errstate(invalid='ignore'): mu=np.nanmean(sub,0); sdv=np.nanstd(sub,0)
        sdv=np.where((sdv==0)|~np.isfinite(sdv),1,sdv); z[ix]=(sub-mu)/sdv
    X=np.hstack([raw,z])
    med=np.nanmedian(X[:ntr],0); med=np.where(np.isfinite(med),med,0)
    inds=np.where(~np.isfinite(X)); X[inds]=np.take(med,inds[1]); X[~np.isfinite(X)]=0
    return X.astype(float)
