functions {
  real prior_lpdf(real x, int fam, real p1, real p2, real p3) {
    // fam codes:
    // 1 beta(a,b)
    // 2 normal(mu,sigma)
    // 3 lognormal(mu,sigma)
    // 4 gamma(shape,rate)
    // 5 exponential(rate)
    // 6 half-normal(sigma)  [implemented as normal(0,sigma) with x constrained >= 0]
    // 7 student_t(df,mu,sigma)
    // 8 cauchy(loc,scale)

    if (fam == 1) return beta_lpdf(x | p1, p2);
    if (fam == 2) return normal_lpdf(x | p1, p2);
    if (fam == 3) return lognormal_lpdf(x | p1, p2);
    if (fam == 4) return gamma_lpdf(x | p1, p2);
    if (fam == 5) return exponential_lpdf(x | p1);
    if (fam == 6) return normal_lpdf(x | 0, p1);
    if (fam == 7) return student_t_lpdf(x | p1, p2, p3);
    if (fam == 8) return cauchy_lpdf(x | p1, p2);
    return negative_infinity();
  }
}
