# Submission form - Kestrel Warranty Claim Review (Variant C)

**1. What did you build, and what business outcome does it move?**
A fraud-risk ranking of claims (logistic regression, partner-history features), a scoring API plus one screen, and a memo. It moves the investigation desk from random checks to targeted ones. June back-test, top 38 claims (= 40/month at 750 claims): 13 frauds found, Rs 18,374 stopped, 25 genuine held (Rs 9,500 goodwill), net about Rs 8,900 = about Rs 234 net per claim checked (Rs 484 gross). Random checking lost about Rs 290 per claim. Wide uncertainty: 22 frauds.

**2. Expected score on hidden outcomes.**
Metric: average precision (PR-AUC) and ROC-AUC, not accuracy (accuracy is meaningless at 1-3% fraud: "flag nothing" gets 96.9% in June). Expected AP about 0.35 (90% interval 0.23-0.54), AUC about 0.92 (0.87-0.97), estimated by training to 31 May 2026 and scoring June, with 500 bootstrap resamples. If test is scored on accuracy after thresholding at top-40/month, expect about 97-98%, no better than flagging nothing. Risk: fraud may spread to new partners after June, which would lower it.

**3. How do you know it works?**
Forward-in-time holdout: 713 June claims, 22 frauds. Top-38 precision 34%; misses are frauds from partners with no history yet (new outlets) and normal-looking inspected claims. Random-split scores would look better and would be misleading. A model trained only on pre-May data scored AUC 0.54 on May-June, so I dropped it.

**4. Did you change, narrow or push back on the ask?**
Pushed back on accuracy as KPI (use rupees stopped per claim checked, as Finance asked). Narrowed "new partners": the signal is a handful of specific partners, not partner age. Decided at build time; reasons are in the memo.

**5. What is wrong with what you are handing us or with the data?**
- 215 open cases are blank in the CRM; older ones were converted to 0 by Zoho and cannot be identified, so some "not fraud" labels in legacy data are really undecided.
- 668 duplicate claim rows (re-submissions, identical apart from a later timestamp) dropped, keeping the first.
- Serials are messy (case, spaces, hyphens) and reused across many claims; I found no signal in them, so they are unused.
- Legacy Zoho resolution events are UTC; none are in the columns provided, so I did not correct anything and did not use timestamps beyond month.
- Test claims are scored with partner history from all training data; the 30-day label lag is not applied to them.
- Only 22 June frauds: all numbers are noisy. The model is not calibrated; score is a ranking.

**6. What did you leave out and why?**
Free-text fields (claim_description, inspector_note): they carried no signal in checks and add risk. Gradient boosting: it lost to logistic regression on June with this little post-May data. Image/photo content: not provided.

**7. Anything built or found that nobody asked for?**
The May auto-approve policy change coincides with fraud rising from about 1% to about 3%, concentrated in auto-approved claims from about 7 partners. Also a second, older pattern: high-value-ratio claims on franchises with repeat claimants.

**8. What did you use AI for?**
Claude (chat) for data analysis, code and drafts. Helpful: fast exploration, service scaffold. Discarded: gradient-boosting model and pre-May training. Cost: no paid API calls. Recording link: [https://www.loom.com/share/62774171a13d4c9bba47c255f47d6c3e]

**Drive link:** https://drive.google.com/drive/folders/1eI347oGm5JKDxIVYKQqLOHgw1Ld-Sw_w?usp=sharing

**9. Someone picks this up Monday and you are unreachable: three things.**
1) Run `python train.py` then `python app.py` (README). 2) Retrain monthly and check the flagged partners still make sense. 3) The score ranks claims; it does not decide payment, and accuracy is not the metric.

**10. Honest hours spent:** 28

**11. GitHub repo:** [https://github.com/harshupadhyay14/Kestrel-Code.git] (data and artifacts are git-ignored per policy s10)

**12. Cost per prediction and per month:** Zero paid calls; local logistic regression. 750 claims x Rs 0 = Rs 0/month. Only hosting cost applies.
