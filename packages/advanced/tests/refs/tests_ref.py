import json, numpy as np, scipy.stats as st
from pathlib import Path
a = [5.1, 4.9, 6.2, 5.8, 6.0, 5.5, 5.3, 6.4, 5.9, 5.0]          # n=10
b = [4.8, 5.2, 5.0, 4.6, 5.1, 4.9, 5.3, 4.7]                    # n=8
c = [6.1, 6.5, 6.3, 6.8, 6.0, 6.6]                              # n=6
before = [200, 195, 210, 188, 205, 199, 202, 190]
after  = [192, 190, 205, 185, 198, 197, 199, 186]
out = {}
r = st.ttest_1samp(a, 5.0)
ci = r.confidence_interval(0.95)
out["ttest1"] = {"x": a, "mu": 5.0, "t": r.statistic, "p": r.pvalue, "df": r.df, "lo": ci.low, "hi": ci.high,
                 "p_less": st.ttest_1samp(a, 5.0, alternative="less").pvalue, "p_greater": st.ttest_1samp(a, 5.0, alternative="greater").pvalue,
                 "ci90_lo": st.ttest_1samp(a, 5.0).confidence_interval(0.90).low, "ci_greater_lo": st.ttest_1samp(a, 5.0, alternative="greater").confidence_interval(0.95).low}
rw = st.ttest_ind(a, b, equal_var=False); rp = st.ttest_ind(a, b, equal_var=True)
out["ttest2"] = {"a": a, "b": b, "welch_t": rw.statistic, "welch_p": rw.pvalue, "welch_df": rw.df, "welch_lo": rw.confidence_interval().low, "welch_hi": rw.confidence_interval().high,
                 "pooled_t": rp.statistic, "pooled_p": rp.pvalue, "pooled_df": rp.df, "pooled_lo": rp.confidence_interval().low, "pooled_hi": rp.confidence_interval().high}
rr = st.ttest_rel(after, before)
out["paired"] = {"after": after, "before": before, "t": rr.statistic, "p": rr.pvalue, "df": rr.df, "lo": rr.confidence_interval().low, "hi": rr.confidence_interval().high}
fa = st.f_oneway(a, b, c)
out["anova"] = {"groups": {"a": a, "b": b, "c": c}, "F": fa.statistic, "p": fa.pvalue}
tab = [[30, 10, 5], [20, 25, 10], [10, 15, 25]]
chi, p, dof, exp = st.chi2_contingency(tab, correction=False)
out["chi2"] = {"table": tab, "stat": chi, "p": p, "df": dof, "expected": exp.tolist()}
tab2 = [[12, 5], [3, 15]]
chi_c, p_c, _, _ = st.chi2_contingency(tab2, correction=True)
chi_n, p_n, _, _ = st.chi2_contingency(tab2, correction=False)
out["chi2_2x2"] = {"table": tab2, "stat_corr": chi_c, "p_corr": p_c, "stat_nocorr": chi_n, "p_nocorr": p_n}
obs = [16, 18, 16, 14, 12, 12]
g = st.chisquare(obs); g2 = st.chisquare(obs, f_exp=[16, 16, 16, 16, 16, 8], ddof=1)
out["gof"] = {"obs": obs, "stat": g.statistic, "p": g.pvalue, "stat2": g2.statistic, "p2": g2.pvalue}
Path("C:/devops/JS Pandas/packages/advanced/tests/fixtures/tests-scipy.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else float(o)))
print("ok")
