import pandas as pd, numpy as np
def load():
    tr=pd.read_csv('train.csv');te=pd.read_csv('test.csv');p=pd.read_csv('partners.csv');pr=pd.read_csv('products.csv')
    tr['is_test']=0;te['is_test']=1
    tr=tr[tr.is_fraud.notna()|(tr.is_test==1)]  # drop undecided (blank) labels
    tr=tr.sort_values('submitted_at').drop_duplicates('claim_id',keep='first')
    return tr,te,p,pr
def build(df,p,pr):
    d=df.merge(p,on='partner_id',how='left').merge(pr,on='sku',how='left')
    d['t']=pd.to_datetime(d.submitted_at)
    d['onb']=pd.to_datetime(d.onboarded_date)
    d['partner_age_d']=(d.t-d.onb).dt.days
    d['new_partner']=(d.partner_age_d<365).astype(int)
    d['uninsp']=(d.partner_inspected=='N').astype(int)
    d['small']=(d.claim_amount_inr<2000).astype(int)
    d['auto_appr']=d.uninsp*d.small
    d['photo']=(d.photo_attached=='Y').astype(int)
    d['ratio']=d.claim_amount_inr/d.list_price_inr
    d['oow']=(d.days_since_purchase>d.warranty_months*30.4).astype(int)
    d['post']=(d.t>='2026-05-01').astype(int)
    d['hour']=d.t.dt.hour
    d['ptype']=d.partner_type.astype('category').cat.codes
    d['legacy']=(d.source=='legacy_zoho').astype(int)
    d['note_na']=d.inspector_note.isna().astype(int)
    d['prior']=d.customer_prior_claims
    return d
F=['partner_age_d','new_partner','uninsp','small','auto_appr','photo','ratio','oow','claim_amount_inr','days_since_purchase','prior','ptype','note_na','list_price_inr','warranty_months']

LAG=pd.Timedelta(days=30)
def add_partner_enc(d,hist,prior_rate=0.012,k=20):
    """Smoothed partner fraud rates from labelled history older than LAG before each row (leak-free)."""
    d=d.copy();h=hist[['partner_id','t','is_fraud','auto_appr']].sort_values('t')
    out_all=[];out_aa=[];out_n=[]
    for pid,g in d.groupby('partner_id'):
        hp=h[h.partner_id==pid];ts=hp.t.values;f=hp.is_fraud.values.cumsum()
        n=np.arange(1,len(hp)+1)
        aa=hp.auto_appr.values;fa=(hp.is_fraud.values*aa).cumsum();na=aa.cumsum()
        idx=np.searchsorted(ts,(g.t-LAG).values,side='left')  # rows strictly older than t-LAG
        def take(arr):
            return np.where(idx>0,arr[np.clip(idx-1,0,max(len(arr)-1,0))] if len(arr) else 0,0)
        F_=take(f) if len(hp) else np.zeros(len(g));N_=take(n) if len(hp) else np.zeros(len(g))
        FA=take(fa) if len(hp) else np.zeros(len(g));NA=take(na) if len(hp) else np.zeros(len(g))
        out_all.append(pd.Series((F_+prior_rate*k)/(N_+k),index=g.index))
        out_aa.append(pd.Series((FA+0.02*k)/(NA+k),index=g.index))
        out_n.append(pd.Series(FA,index=g.index))
    d['p_rate']=pd.concat(out_all);d['p_aa_rate']=pd.concat(out_aa);d['p_aa_frauds']=pd.concat(out_n)
    return d
F2=F+['p_rate','p_aa_rate','p_aa_frauds']
