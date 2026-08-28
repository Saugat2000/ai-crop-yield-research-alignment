"""Phase 11 — model scaffolds for the research-side outcomes.

Four outcomes, four families, each chosen for a property of the outcome rather than for
fit:

    participation      does a country-crop system have ANY eligible study?   logit/probit
    intensity          how many studies, fractionally counted?               Poisson/NB
    zero-inflation     are the zeros and the counts generated differently?   hurdle
    local leadership   what SHARE of a system's studies are locally led?     fractional logit
    citation uptake    citations per study, heavily over-dispersed           NB

Every function returns a fitted statsmodels result plus a diagnostics dictionary, so a
specification is never reported without the checks that qualify it.

**No substantive model may be estimated before Gate G1.** `fit_participation` and its
siblings are pure functions of the data handed to them and are exercised here on
synthetic data with known parameters; the pipeline that feeds them real data calls
`require_search_complete` first.

Counting rule: fractional counting is primary for intensity (CLAUDE.md §5.1). A study of
three countries contributes 1/3 to each. Full counts are carried alongside for
descriptive comparison and every reported number states which rule it uses.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def _design(df: pd.DataFrame, x_cols, add_const=True):
    X = df[list(x_cols)].astype(float)
    return sm.add_constant(X, has_constant="add") if add_const else X


def _vif(X: pd.DataFrame) -> dict:
    """Variance inflation, computed without statsmodels' outliers_influence import cost."""
    out = {}
    cols = [c for c in X.columns if c != "const"]
    for c in cols:
        others = [o for o in cols if o != c]
        if not others:
            out[c] = 1.0
            continue
        y = X[c].to_numpy()
        A = sm.add_constant(X[others].to_numpy(), has_constant="add")
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out[c] = float("inf") if r2 >= 1 else 1.0 / (1.0 - r2)
    return out


# ---------------------------------------------------------------------------
# participation: binary
# ---------------------------------------------------------------------------

def fit_participation(df, y_col, x_cols, *, link="logit", cluster_col=None):
    """Logit or probit for whether a country-crop system is studied at all.

    Cluster-robust standard errors by country are the default in use, because
    country-crop rows within a country share unobserved capacity.
    """
    y = df[y_col].astype(float)
    X = _design(df, x_cols)
    fam = sm.families.Binomial(sm.families.links.Logit() if link == "logit"
                               else sm.families.links.Probit())
    model = sm.GLM(y, X, family=fam)
    if cluster_col is not None:
        res = model.fit(cov_type="cluster",
                        cov_kwds={"groups": df[cluster_col].to_numpy()})
    else:
        res = model.fit()

    p = np.clip(res.fittedvalues, 1e-9, 1 - 1e-9)
    diag = {
        "link": link,
        "n": int(len(y)),
        "events": int(y.sum()),
        "events_per_variable": float(min(y.sum(), (1 - y).sum()) / max(1, len(x_cols))),
        "separation_suspected": bool((p < 1e-6).any() or (p > 1 - 1e-6).any()),
        "vif": _vif(X),
        "converged": bool(res.converged) if hasattr(res, "converged") else True,
        "pseudo_r2": float(1 - res.deviance / res.null_deviance)
        if getattr(res, "null_deviance", 0) else float("nan"),
    }
    if diag["events_per_variable"] < 10:
        diag["warning"] = (f"{diag['events_per_variable']:.1f} events per variable is "
                           f"below the conventional 10; coefficients are unstable")
    return res, diag


def marginal_effects(res, X: pd.DataFrame, *, at="mean"):
    """Average marginal effects on the probability scale.

    Coefficients from a logit are log-odds and are not comparable across models or
    interpretable as a probability change, so a binary result is always reported as an
    average marginal effect.
    """
    try:
        me = res.get_margeff(at=at) if hasattr(res, "get_margeff") else None
        if me is not None:
            return pd.DataFrame({"dydx": me.margeff, "se": me.margeff_se},
                                index=[c for c in X.columns if c != "const"])
    except (NotImplementedError, ValueError):
        pass
    # GLM has no get_margeff: differentiate numerically.
    base = res.predict(X)
    out = {}
    for c in X.columns:
        if c == "const":
            continue
        h = 0.01 * (X[c].std() or 1.0)
        Xp = X.copy(); Xp[c] = Xp[c] + h
        out[c] = float((res.predict(Xp) - base).mean() / h)
    return pd.DataFrame({"dydx": pd.Series(out)})


# ---------------------------------------------------------------------------
# intensity: counts
# ---------------------------------------------------------------------------

def fit_count(df, y_col, x_cols, *, family="poisson", exposure_col=None,
              cluster_col=None):
    """Poisson (PPML) or negative binomial for research intensity.

    Poisson pseudo-maximum likelihood is consistent under mean correctness even when the
    variance is misspecified, so it is the honest default with robust errors; the
    negative binomial is fitted alongside because over-dispersion is expected and the
    comparison is itself reportable.

    Fractional counts are non-integer. That is fine for Poisson/PPML, which needs only a
    correct conditional mean, and it is why PPML rather than a count likelihood requiring
    integers is the primary specification.
    """
    y = df[y_col].astype(float)
    X = _design(df, x_cols)
    offset = np.log(df[exposure_col].astype(float)) if exposure_col else None

    if family == "poisson":
        model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    elif family == "negbin":
        if (y % 1 != 0).any():
            warnings.warn("negative binomial expects integer counts; fractional counts "
                          "are being rounded for this comparison specification only")
            y = np.rint(y)
        model = sm.NegativeBinomial(y, X, offset=offset)
    else:
        raise ValueError(f"unknown family {family!r}")

    if cluster_col is not None and family == "poisson":
        res = model.fit(cov_type="cluster",
                        cov_kwds={"groups": df[cluster_col].to_numpy()})
    else:
        res = model.fit(disp=0) if family == "negbin" else model.fit()

    mu = np.asarray(res.fittedvalues, dtype=float)
    resid = np.asarray(y, dtype=float) - mu
    dof = max(1, len(y) - X.shape[1])
    pearson_chi2 = float((resid ** 2 / np.clip(mu, 1e-9, None)).sum())
    diag = {
        "family": family,
        "n": int(len(y)),
        "zero_share": float((np.asarray(y) == 0).mean()),
        "mean": float(np.mean(y)),
        "variance": float(np.var(y)),
        "dispersion_ratio": float(np.var(y) / np.mean(y)) if np.mean(y) > 0 else float("nan"),
        "pearson_dispersion": pearson_chi2 / dof,
        "vif": _vif(X),
        "fractional_counts_present": bool((np.asarray(y) % 1 != 0).any()),
    }
    diag["overdispersed"] = diag["pearson_dispersion"] > 1.5
    if diag["overdispersed"] and family == "poisson":
        diag["note"] = ("Pearson dispersion above 1.5: report PPML with robust errors and "
                        "compare against the negative binomial")
    return res, diag


def overdispersion_test(res_poisson, y) -> dict:
    """Cameron-Trivedi regression-based test for Var = mu + alpha*mu^2."""
    mu = np.asarray(res_poisson.fittedvalues, dtype=float)
    y = np.asarray(y, dtype=float)
    z = ((y - mu) ** 2 - y) / np.clip(mu, 1e-9, None)
    A = sm.add_constant(mu)
    aux = sm.OLS(z, A).fit()
    return {"alpha": float(aux.params[1]), "t": float(aux.tvalues[1]),
            "p_value": float(aux.pvalues[1]),
            "overdispersed": bool(aux.pvalues[1] < 0.05 and aux.params[1] > 0)}


# ---------------------------------------------------------------------------
# hurdle
# ---------------------------------------------------------------------------

def fit_hurdle(df, y_col, x_cols_participation, x_cols_intensity, *, cluster_col=None):
    """Two-part hurdle: whether a system is studied, then how much given that it is.

    Preferred over zero-inflation here because the zeros are structural in a specific
    sense: a country-crop system with no eligible study is a real zero (CLAUDE.md §3.7),
    not a count that happened to come out zero. The two parts answer different questions
    and are reported separately.
    """
    y = df[y_col].astype(float)
    part = (y > 0).astype(float)

    d1 = df.copy(); d1["_participates"] = part
    res1, diag1 = fit_participation(d1, "_participates", x_cols_participation,
                                    cluster_col=cluster_col)

    pos = df[y > 0].copy()
    if len(pos) < len(x_cols_intensity) + 2:
        raise ValueError(f"only {len(pos)} positive observations; too few to fit the "
                         f"intensity part on {len(x_cols_intensity)} predictors")
    res2, diag2 = fit_count(pos, y_col, x_cols_intensity, family="poisson",
                            cluster_col=cluster_col)

    return {"participation": res1, "intensity": res2}, {
        "participation": diag1, "intensity": diag2,
        "n_total": int(len(y)), "n_positive": int((y > 0).sum()),
        "zero_share": float((y == 0).mean()),
    }


# ---------------------------------------------------------------------------
# fractional: shares
# ---------------------------------------------------------------------------

def fit_fractional(df, y_col, x_cols, *, cluster_col=None):
    """Fractional logit (Papke-Wooldridge) for a share bounded in [0, 1].

    Local leadership is a proportion that legitimately equals exactly 0 or exactly 1, so
    a log-odds transform is undefined and OLS predicts outside the unit interval. A
    quasi-likelihood GLM with a logit link and robust errors handles both.
    """
    y = df[y_col].astype(float)
    if (y < 0).any() or (y > 1).any():
        raise ValueError("fractional outcome must lie in [0, 1]")
    X = _design(df, x_cols)
    model = sm.GLM(y, X, family=sm.families.Binomial())
    if cluster_col is not None:
        res = model.fit(cov_type="cluster",
                        cov_kwds={"groups": df[cluster_col].to_numpy()})
    else:
        res = model.fit(cov_type="HC1")
    diag = {
        "n": int(len(y)),
        "boundary_zero_share": float((y == 0).mean()),
        "boundary_one_share": float((y == 1).mean()),
        "mean_outcome": float(y.mean()),
        "vif": _vif(X),
        "predictions_within_unit_interval": bool(
            (res.fittedvalues >= 0).all() and (res.fittedvalues <= 1).all()),
    }
    return res, diag


# ---------------------------------------------------------------------------
# residual spatial autocorrelation
# ---------------------------------------------------------------------------

def residual_moran(residuals, W, *, permutations=999, seed=20260805) -> dict:
    """Moran's I on model residuals, with a permutation reference distribution.

    Row-standardised W. Reported for every spatial specification: residual clustering
    means the model has not absorbed the spatial structure and a non-spatial standard
    error is too small.
    """
    r = np.asarray(residuals, dtype=float)
    r = r - r.mean()
    Wm = np.asarray(W, dtype=float)
    n = len(r)
    s0 = Wm.sum()
    if s0 == 0 or np.allclose(r, 0):
        return {"I": float("nan"), "p_value": float("nan"), "n": n}
    num = float(r @ Wm @ r)
    den = float(r @ r)
    I = (n / s0) * (num / den)

    rng = np.random.default_rng(seed)
    perm = np.empty(permutations)
    for i in range(permutations):
        rp = rng.permutation(r)
        perm[i] = (n / s0) * float(rp @ Wm @ rp) / float(rp @ rp)
    p = float((np.sum(np.abs(perm) >= abs(I)) + 1) / (permutations + 1))
    return {"I": float(I), "expected_I": -1.0 / (n - 1), "p_value": p,
            "permutations": permutations, "n": n,
            "residual_clustering": bool(p < 0.05)}


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def compare_models(fitted: dict) -> pd.DataFrame:
    """One table of every specification, so a preferred model is never shown alone."""
    rows = []
    for name, res in fitted.items():
        rows.append({
            "model": name,
            "n": int(res.nobs),
            "loglik": float(getattr(res, "llf", float("nan"))),
            "aic": float(getattr(res, "aic", float("nan"))),
            "bic": float(getattr(res, "bic", float("nan"))),
            "df_model": float(getattr(res, "df_model", float("nan"))),
        })
    return pd.DataFrame(rows).sort_values("aic")


def add_fixed_effects(df, x_cols, fe_cols, *, drop_first=True, min_group=2):
    """Add region and crop fixed effects, dropping groups too small to identify.

    A group with one observation contributes a dummy that fits it perfectly and removes it
    from identification. Dropping such groups explicitly, and reporting which, is honest;
    leaving them in produces a silently rank-deficient design.
    """
    d = df.copy()
    kept, dropped = [], {}
    for c in fe_cols:
        counts = d[c].value_counts()
        small = counts[counts < min_group].index.tolist()
        if small:
            dropped[c] = small
            d = d[~d[c].isin(small)]
        dummies = pd.get_dummies(d[c], prefix=c, drop_first=drop_first, dtype=float)
        d = pd.concat([d, dummies], axis=1)
        kept.extend(dummies.columns.tolist())
    return d, list(x_cols) + kept, {"dropped_small_groups": dropped,
                                    "n_fe_columns": len(kept),
                                    "n_rows_after": int(len(d))}


def predict_holdout(res, df_new, x_cols, *, exposure_col=None):
    """Predict on held-out data, refusing silently-misaligned inputs."""
    missing = [c for c in x_cols if c not in df_new.columns]
    if missing:
        raise KeyError(f"held-out data is missing predictors: {missing}")
    X = _design(df_new, x_cols)
    train_cols = list(getattr(res.model, "exog_names", X.columns))
    if list(X.columns) != train_cols:
        X = X.reindex(columns=train_cols, fill_value=0.0)
    pred = res.predict(X)
    if exposure_col is not None:
        pred = np.asarray(pred) * np.asarray(df_new[exposure_col], dtype=float)
    return np.asarray(pred, dtype=float)


def check_estimability(df, y_col, x_cols) -> dict:
    """Refuse-or-warn checks run BEFORE fitting, so a failure is diagnosed not crashed.

    Catches the conditions that make a fit meaningless rather than merely imperfect:
    missing covariates, zero variance, a rank-deficient design, no variation in the
    outcome, and high-leverage rows.
    """
    problems, warnings_ = [], []
    missing = [c for c in list(x_cols) + [y_col] if c not in df.columns]
    if missing:
        problems.append(f"columns absent: {missing}")
        return {"estimable": False, "problems": problems, "warnings": warnings_}

    sub = df[list(x_cols) + [y_col]]
    n_missing = int(sub.isna().any(axis=1).sum())
    if n_missing:
        warnings_.append(f"{n_missing} rows have a missing value and would be dropped")

    d = sub.dropna()
    if len(d) <= len(x_cols) + 1:
        problems.append(f"{len(d)} complete rows for {len(x_cols)} predictors")
        return {"estimable": False, "problems": problems, "warnings": warnings_,
                "n_complete": int(len(d))}

    y = d[y_col]
    if y.nunique() < 2:
        problems.append("the outcome does not vary")

    const_cols = [c for c in x_cols if d[c].nunique() < 2]
    if const_cols:
        problems.append(f"predictors with no variation: {const_cols}")

    X = _design(d, x_cols).to_numpy(dtype=float)
    rank = int(np.linalg.matrix_rank(X))
    if rank < X.shape[1]:
        problems.append(f"design matrix is rank deficient: rank {rank} of {X.shape[1]}")

    lev = None
    if rank == X.shape[1]:
        try:
            H = X @ np.linalg.solve(X.T @ X, X.T)
            lev = np.diag(H)
            thresh = 3 * X.shape[1] / len(d)
            n_high = int((lev > thresh).sum())
            if n_high:
                warnings_.append(f"{n_high} rows exceed 3p/n leverage")
        except np.linalg.LinAlgError:
            problems.append("could not invert X'X")

    return {
        "estimable": not problems, "problems": problems, "warnings": warnings_,
        "n_complete": int(len(d)), "n_dropped_missing": n_missing,
        "rank": rank, "n_parameters": int(X.shape[1]),
        "max_leverage": float(lev.max()) if lev is not None else float("nan"),
    }


def publication_table(fitted: dict, *, digits=3, marginal=None) -> pd.DataFrame:
    """One publication-ready table of coefficients across specifications.

    Reports the estimate, its standard error and interval. It deliberately does NOT print
    significance stars: CLAUDE.md 5.6 reserves "significant" for a formal result stated
    with an estimate and an interval, which is what this table gives.
    """
    rows = []
    for name, res in fitted.items():
        params, bse = res.params, res.bse
        try:
            ci = res.conf_int()
            lo = ci.iloc[:, 0] if hasattr(ci, "iloc") else ci[:, 0]
            hi = ci.iloc[:, 1] if hasattr(ci, "iloc") else ci[:, 1]
        except Exception:
            lo = hi = pd.Series(np.nan, index=params.index)
        for term in params.index:
            rows.append({
                "model": name, "term": term,
                "estimate": round(float(params[term]), digits),
                "std_error": round(float(bse[term]), digits),
                "ci_low": round(float(lo[term]), digits),
                "ci_high": round(float(hi[term]), digits),
                "n": int(res.nobs),
                "cov_type": getattr(res, "cov_type", "nonrobust"),
            })
    out = pd.DataFrame(rows)
    out.attrs["note"] = ("Estimates with standard errors and 95% intervals. No "
                         "significance stars: an interval states the evidence.")
    return out


def vuong_test(res_a, res_b, y) -> dict:
    """Vuong test for two non-nested models on the same data."""
    ll_a = res_a.model.loglikeobs(res_a.params) if hasattr(res_a.model, "loglikeobs") else None
    ll_b = res_b.model.loglikeobs(res_b.params) if hasattr(res_b.model, "loglikeobs") else None
    if ll_a is None or ll_b is None:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "note": "loglikeobs unavailable for one model"}
    m = np.asarray(ll_a) - np.asarray(ll_b)
    n = len(m)
    stat = float(np.sqrt(n) * m.mean() / (m.std(ddof=1) or 1e-12))
    return {"statistic": stat, "p_value": float(2 * (1 - stats.norm.cdf(abs(stat)))),
            "favours": "A" if stat > 0 else "B"}
