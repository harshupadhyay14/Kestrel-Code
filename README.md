# Kestrel claim review

Scores warranty claims for fraud risk so the 40-claims/month investigation desk looks at the right ones first.

## Run (clean machine, no API key, no network)
```
pip install -r requirements.txt
# put train.csv, test_unlabelled.csv, partners.csv, products.csv in ./data  (NOT committed: policy s10 forbids publishing customer data)
python train.py      # cleans, validates, writes predictions.csv + artifacts/
python app.py        # screen at http://localhost:5000, API: POST /score
```
```
curl -X POST localhost:5000/score -H 'content-type: application/json' -d '{"claim_id":"C1","submitted_at":"2026-07-15 11:20","partner_id":"SP3160","sku":"KH-AF-03","product_serial":"KH123456789","days_since_purchase":120,"claim_amount_inr":1400,"photo_attached":"N","partner_inspected":"N","customer_prior_claims":1}'
```
Bad input returns HTTP 400 with a plain message. The model is a small logistic regression; no paid calls anywhere.

## Method in short
- Dropped 215 undecided (blank) labels and 668 re-submitted duplicate claim rows before training/validation.
- Validation is forward in time (train to 31 May 2026, score June), because the 1 May auto-approve change made a new fraud regime. A model trained only on pre-May data scores AUC 0.54 on May-June: the old patterns do not carry over.
- Signals: small claim auto-approved without inspection, partner's own fraud history (leak-free, 30-day label lag), claim/list-price ratio, customer prior claims. "New partner" alone is weak; specific partners matter.
- Results: `artifacts/evidence.json` (created by train.py).

## Known limits
Only 22 frauds in the June holdout; intervals are wide. Partner history goes stale: rerun train.py monthly. Score is a ranking, not a calibrated probability.
