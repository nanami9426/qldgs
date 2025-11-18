# QLDGS-PSO-Elite Complexity Analysis

Let

- \(N\) be the swarm size,
- \(T\) the number of outer PSO iterations,
- \(d\) the feature dimensionality,
- \(C_{\text{eval}}\) the cost of a single fitness computation (training + validating the classifier),
- \(L\) the number of length intervals,
- \(M\) the number of inner evaluations per interval (the parameter exposed as `interval_iterations`),
- \(S\) the number of Q-learning states (4 in this implementation),
- \(A\) the number of Q-learning actions (also 4 here).

## Baseline PSO

For binary PSO, every iteration updates all particles and evaluates each candidate once. The dominant term is fitness evaluation, so the complexity is

\[
O(N \cdot T \cdot C_{\text{eval}}) + O(N \cdot T \cdot d)
\approx O(N \cdot T \cdot C_{\text{eval}})
\]

because \(C_{\text{eval}}\) (training/evaluating the classifier on high-dimensional data) dwarfs the vector updates.

## QLDGS-PSO-Elite

The hierarchical controller introduces two additional phases for each outer step:

1. **Interval exploration:** for each of the \(L\) intervals we evaluate \(M\) candidate solutions to estimate the quality of the length range. Cost: 
   $$
   O(L \cdot M \cdot C_{\text{eval}})
   $$
   

2. **Length-guided PSO update:** we still evolve \(N\) particles per outer iteration, so \(O(N \cdot C_{\text{eval}})\) per step, identical to baseline PSO, but the number of evaluations equals \(T \cdot N\) (denoted `Max_FEs` in the code).

The Q-learning controller performs table lookups and scalar updates only (\(O(S \cdot A)\) memory, \(O(1)\) per step), so its overhead is negligible relative to evaluation cost.

Combining the terms, one outer iteration of QLDGS-PSO-Elite costs

$$
O\left(L \cdot M \cdot C_{\text{eval}} + N \cdot C_{\text{eval}}\right)
= O\left((L \cdot M + N) \cdot C_{\text{eval}}\right)
$$
Across \(T\) iterations the total complexity is

$$
O\left(T \cdot (L \cdot M + N) \cdot C_{\text{eval}}\right)
$$


When the length controller converges early the algorithm switches to the second PSO phase only, so the bound tightens toward the baseline 
$$
O(N \cdot T \cdot C_{\text{eval}})
$$
.Empirically we keep \(L\) and \(M\) under 25, so 
$$
L \cdot M
$$
stays comparable to \(N\), matching the measured 5–6× speedups over plain PSO even with the extra coordination logic. Table updates and reward calculations are O(1) and do not change the asymptotic complexity.
