"""Train, validate, write predictions. Usage: python train.py  (expects the 4 CSVs in ./data)"""
import json, pickle, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
import kestrel_feats as K
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
def rd(n): return pd.read_csv(f'data/{n}.csv')
tr=rd('train');te=rd('test_unlabelled');p=rd('partners');pr=rd('products')
tr['is_test']=0;te['is_test']=1
n_raw=len(tr);n_undec=int(tr.is_fraud.isna().sum())
tr=tr[tr.is_fraud.notna()].sort_values('submitted_at')
n_dup=int(tr.claim_id.duplicated().sum())
tr=tr.drop_duplicates('claim_id',keep='first')
d=K.add_partner_enc(K.build(tr,p,pr),K.build(tr,p,pr))
FEATS=['aa','aa_pr','aa_new','hi_ratio','prior','franch','lpr','la']
LABELS={'aa':'small claim auto-approved without inspection','aa_pr':'partner\'s track record on auto-approved claims',
 'aa_new':'auto-approved claim from a partner onboarded <12 months ago','hi_ratio':'claim is >46% of the product list price',
 'prior':'customer\'s earlier claims','franch':'partner is a franchise','lpr':'partner\'s overall fraud history','la':'claim amount'}
def X(d):
    x=pd.DataFrame(index=d.index)
    x['aa']=d.auto_appr;x['aa_pr']=d.auto_appr*np.log(d.p_aa_rate/0.02+1e-9);x['aa_new']=d.auto_appr*d.new_partner
    x['hi_ratio']=(d.ratio>0.46).astype(int);x['prior']=d.prior.clip(upper=4)
    x['franch']=(d.partner_type=='franchise').astype(int);x['lpr']=np.log(d.p_rate/0.012);x['la']=np.log(d.claim_amount_inr)
    return x[FEATS]
def mk(): return make_pipeline(StandardScaler(),LogisticRegression(C=0.1,class_weight='balanced',max_iter=2000))
GOODWILL=380
def econ(y,s,amt,k):
    o=np.argsort(-s)[:k];caught=(y.values[o]*amt.values[o]).sum();fp=(1-y.values[o]).sum()
    return dict(k=k,fraud_found=int(y.values[o].sum()),rs_stopped=float(caught),genuine_held=int(fp),net_rs=float(caught-GOODWILL*fp))
# ---- forward validation: train <=31 May 2026, score June 2026 (regime after the 1 May auto-approve change)
fit=d[d.t<'2026-06-01'];val=d[(d.t>='2026-06-01')&(d.t<'2026-07-01')]
m=mk().fit(X(fit),fit.is_fraud);s=m.predict_proba(X(val))[:,1]
k=int(round(len(val)*40/750));rng=np.random.default_rng(0);aps=[];aucs=[]
for _ in range(500):
    i=rng.integers(0,len(val),len(val));y=val.is_fraud.values[i]
    if y.sum()>0: aps.append(ap(y,s[i]));aucs.append(auc(y,s[i]))
rnd=[]
for _ in range(500):
    o=rng.permutation(len(val))[:k];rnd.append((val.is_fraud.values[o]*val.claim_amount_inr.values[o]).sum()-GOODWILL*(1-val.is_fraud.values[o]).sum())
acc_all_no=1-val.is_fraud.mean()
thr=np.sort(s)[::-1][k-1];acc_flag=((s>=thr)==val.is_fraud.astype(bool)).mean()
m1=mk().fit(X(d[d.t<'2026-05-01']),d[d.t<'2026-05-01'].is_fraud);v2=d[d.t>='2026-05-01'];s2=m1.predict_proba(X(v2))[:,1]
rep=dict(n_train_raw=n_raw,undecided_dropped=n_undec,duplicate_claim_rows_dropped=n_dup,n_train_used=len(d),fraud_rate=float(d.is_fraud.mean()),
 june=dict(n=len(val),frauds=int(val.is_fraud.sum()),AP=float(ap(val.is_fraud,s)),AP_ci=[float(np.percentile(aps,5)),float(np.percentile(aps,95))],
   AUC=float(auc(val.is_fraud,s)),AUC_ci=[float(np.percentile(aucs,5)),float(np.percentile(aucs,95))],econ_top_k=econ(val.is_fraud,s,val.claim_amount_inr,k),
   random_review_net_rs_mean=float(np.mean(rnd)),accuracy_predict_all_genuine=float(acc_all_no),accuracy_of_flag_top_k=float(acc_flag),base_rate=float(val.is_fraud.mean())),
 shift_check_train_before_May_score_May_Jun=dict(AUC=float(auc(v2.is_fraud,s2)),AP=float(ap(v2.is_fraud,s2))))
# ---- final model on everything, partner encodings from all labelled history
final=mk().fit(X(d),d.is_fraud)
tt=K.build(te,p,pr)
# for test rows, use all labelled history (train ends 30 Jun; lag rule would zero it out for later months)
K.LAG=pd.Timedelta(days=0);tt=K.add_partner_enc(tt,d)
sc=final.predict_proba(X(tt))[:,1]
sub=pd.DataFrame({'claim_id':te.claim_id.values,'score':np.round(sc,6)});sub.to_csv('predictions.csv',index=False)
rep['test']=dict(n=len(sub),mean_score=float(sc.mean()),n_auto_approved=int(tt.auto_appr.sum()))
# ---- artifacts for the service
hist=d.groupby('partner_id').agg(n=('is_fraud','size'),f=('is_fraud','sum')).to_dict('index')
ha=d[d.auto_appr==1].groupby('partner_id').agg(n=('is_fraud','size'),f=('is_fraud','sum')).to_dict('index')
sc_=final.named_steps['standardscaler'];lr=final.named_steps['logisticregression']
pickle.dump(dict(model=final,hist=hist,hist_aa=ha,feats=FEATS,labels=LABELS,mean=sc_.mean_,scale=sc_.scale_,coef=lr.coef_[0],
   score_threshold=float(np.sort(sc)[::-1][min(len(sc)-1,int(len(sc)*40/750))]),
   ),open('artifacts/model.pkl','wb'))
json.dump(rep,open('artifacts/evidence.json','w'),indent=1)
print(json.dumps(rep,indent=1))
