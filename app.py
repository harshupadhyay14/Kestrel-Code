"""Kestrel claim-review service. POST /score (JSON) -> score + reasons. GET / -> screen. No API key, no network."""
import pickle, os, numpy as np, pandas as pd
from flask import Flask, request, jsonify, render_template_string
import kestrel_feats as K
os.chdir(os.path.dirname(os.path.abspath(__file__)))
A=pickle.load(open('artifacts/model.pkl','rb'))
P=pd.read_csv('data/partners.csv');PR=pd.read_csv('data/products.csv')
app=Flask(__name__)
REQ=['claim_id','submitted_at','partner_id','sku','product_serial','days_since_purchase','claim_amount_inr','photo_attached','partner_inspected','customer_prior_claims']
def score(rec):
    miss=[c for c in REQ if c not in rec or rec[c] in (None,'')]
    if miss: raise ValueError('missing fields: '+', '.join(miss))
    if rec['partner_id'] not in set(P.partner_id): raise ValueError('unknown partner_id '+str(rec['partner_id']))
    if rec['sku'] not in set(PR.sku): raise ValueError('unknown sku '+str(rec['sku']))
    df=pd.DataFrame([{**rec,'inspector_note':rec.get('inspector_note'),'source':'crm'}])
    for c in ['days_since_purchase','claim_amount_inr','customer_prior_claims']: df[c]=pd.to_numeric(df[c])
    d=K.build(df,P,PR)
    h=A['hist'].get(rec['partner_id'],{'n':0,'f':0});ha=A['hist_aa'].get(rec['partner_id'],{'n':0,'f':0})
    d['p_rate']=(h['f']+0.012*20)/(h['n']+20);d['p_aa_rate']=(ha['f']+0.02*20)/(ha['n']+20)
    r=d.iloc[0]
    x={'aa':r.auto_appr,'aa_pr':r.auto_appr*np.log(r.p_aa_rate/0.02+1e-9),'aa_new':r.auto_appr*r.new_partner,'hi_ratio':float(r.ratio>0.46),
       'prior':min(r.prior,4),'franch':float(r.partner_type=='franchise'),'lpr':np.log(r.p_rate/0.012),'la':np.log(r.claim_amount_inr)}
    xv=np.array([x[f] for f in A['feats']],float);z=(xv-A['mean'])/A['scale'];contrib=z*A['coef']
    prob=float(A['model'].predict_proba(xv.reshape(1,-1))[0,1])
    order=np.argsort(-contrib);reasons=[]
    for i in order[:3]:
        if contrib[i]>0.05: reasons.append(A['labels'][A['feats'][i]])
    if not reasons: reasons=['no strong risk signals']
    facts=[f"Partner {rec['partner_id']} ({r.city}, onboarded {r.onboarded_date}, {r.partner_type.replace('_',' ')})",
           f"Amount Rs {r.claim_amount_inr:,.0f} = {r.ratio:.0%} of list price; inspection: {rec['partner_inspected']}",
           f"Partner history: {int(h['f'])} confirmed fraud in {int(h['n'])} past claims ({int(ha['f'])} in {int(ha['n'])} auto-approved)"]
    return dict(claim_id=rec['claim_id'],score=round(prob,4),flag_for_review=bool(prob>=A['score_threshold']),reasons=reasons,context=facts,
                note='Score is a ranking, not a probability. Investigation desk capacity is 40 claims/month: review highest scores first.')
@app.post('/score')
def api():
    try: return jsonify(score(request.get_json(force=True)))
    except (ValueError,KeyError) as e: return jsonify(error=str(e)),400
    except Exception as e: return jsonify(error='could not score this record: '+str(e)),500
PAGE="""<!doctype html><meta name=viewport content="width=device-width"><title>Kestrel claim check</title>
<style>body{font-family:system-ui;max-width:640px;margin:2em auto;padding:0 1em}label{display:block;margin:.5em 0 .1em;font-size:.85em;color:#555}input,select{width:100%;padding:.4em}
button{margin-top:1em;padding:.6em 1.2em}.r{margin-top:1.5em;padding:1em;border-radius:8px;background:#f3f3f3}.f{background:#fde8e8}</style>
<h2>Warranty claim check</h2><form id=f>
{% for k,v in fields %}<label>{{k}}</label><input name={{k}} value="{{v}}">{% endfor %}
<button>Score claim</button></form><div id=o></div>
<script>f.onsubmit=async e=>{e.preventDefault();const b=Object.fromEntries(new FormData(f));const r=await fetch('/score',{method:'POST',body:JSON.stringify(b)});const j=await r.json();
o.innerHTML=j.error?'<div class=r>'+j.error+'</div>':`<div class="r ${j.flag_for_review?'f':''}"><b>${j.flag_for_review?'REVIEW BEFORE PAYING':'OK to process'}</b> - risk score ${j.score}<ul>${j.reasons.map(x=>'<li>'+x+'</li>').join('')}</ul><small>${j.context.join('<br>')}</small><p><small>${j.note}</small></p></div>`}</script>"""
EX=[('claim_id','DEMO1'),('submitted_at','2026-07-15 11:20'),('partner_id','SP3160'),('sku','KH-AF-03'),('product_serial','KH123456789'),('days_since_purchase','120'),('claim_amount_inr','1400'),('photo_attached','N'),('partner_inspected','N'),('customer_prior_claims','1')]
@app.get('/')
def home(): return render_template_string(PAGE,fields=EX)
if __name__=='__main__': app.run(host='0.0.0.0',port=int(os.environ.get('PORT',5000)))
