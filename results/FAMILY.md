# The (p, d) family — first round (2026-10-05)

Generalisation of the c_e paper (case (p, d) = (2, 3)) to the split graded model of every degree-d cyclic division
algebra over F_p((t)). Labels: [Proved], [Verified] (exact computation), [Conjecture], [Retracted].

## 0. Setup

* A^s = F_p^d[φ; s] (s = cyclic shift of the idempotents e_0..e_{d−1}), t = (1,…,1)φ^d central,
  L = L_{p,d} = A^s_{>0} / tF_p[t] with commutator bracket and p-map x ↦ x^p. Letters y_{k,i} = e_iφ^k
  (for d | k only i ≤ d−2). [e_aφ^i, e_bφ^j] = (δ_{a,b+i}e_a − δ_{b,a+j}e_b)φ^{i+j}; (e_rφ^k)^{[p]} = [d|k] e_rφ^{pk}.
  L ⊗ F_{p^d} is gr of P = (1+φO)/(1+tF_p[[t]]), O = F_{p^d}[[φ;σ]], φ^d = t (Prop. split of the paper, verbatim).
  For p ∤ d, L is the positive part of A_{d−1}^{(1)} in the principal grading; for p | d it is the pgl_d-version
  (the scalar loop t^k I is not split off), see §2.
* U = u(L), u_m = u(L/L_{≥m}), d_k = d (d ∤ k) or d − 1 (d | k), dim u_m = p^{N_m}.
* σ' = σ'_{p,d} = Σ_{r ∈ Z/d} e_r e_{r+1} ⋯ e_{r+d−1} (e_i = y_{1,i}), the cyclic sum of the Coxeter word ("orient +1");
  orient −1 is the reversed word. For (2,3) this is exactly the leading form of F (paper, Thm 7.1 step (1)).
* Geometric meaning: these algebras are the local (pro-p) side of the Lubotzky–Samuels–Vishne Ramanujan complexes
  of type Ã_{d−1}, which are built from cyclic division algebras over F_q(t) [reference to be checked]. For other
  (p, d) we have NOT constructed the lattice or the group-level Hecke element F. Everything below is about the graded
  model with σ' as the candidate leading form.

Code: `fam.py` (generic PBW engine over F_p, exponents < p, Q-blocks, rank mod p), `validate23.py`, `sweep.py`,
`colideal.py`, `obstruction.py`, `uinj.py`. Outputs: `log_p*_d*.txt`, `out_sweep_*.txt`.

**Validation [Verified].** At (2,3): σ'(+1) equals the paper's σ' term by term; the total kernel of R_σ' on u_m is
7, 37, 100, 547, 3182, 10226 and k_m = 1, 3, 6, 10, 15, 21 for m = 2..7, exactly the paper's table.

## 1. Non-zero-divisor theorem for all p ∤ d  [Proved]

**Theorem 1.** Let d ≥ 2 and p ∤ d. Then left and right multiplication by σ'_{p,d} are injective on U = u(L_{p,d}).
Hence k_m ≥ m − d for all m.

*Proof* (the char-2 proof of the paper, with the two points that need p ∤ d made explicit).
(1) π: A^s → M_d(F_p[t]), e_i ↦ E_ii, φ ↦ Π = Σ_i E_{i+1,i} + tE_{0,d−1}, is an injective graded algebra map
(Π E_ii = E_{i+1,i+1}Π, Π^d = tI).
(2) ρ(x) := π(x) − (tr π(x)/d)·I kills T (π(t^k) = t^kI, tr = d t^k) and is a restricted representation of L:
(π(x) − cI)^p = π(x)^p − c^pI, and tr(X^p) = (tr X)^p in characteristic p, d^p ≡ d (mod p). Here p ∤ d is used.
ρ is injective on L (ker π|_{A_{>0}} + scalars ∩ image = T).
(3) Dual representation ρ*(x) = −ρ(x)^T. tr π(e_iφ) = 0, so ρ*(e_i) = −E_{i−1,i} (i ≥ 1), ρ*(e_0) = −tE_{d−1,0}. A
cyclic word e_r e_{r+1}⋯e_{r+d−1} maps to (−1)^d t E_{r−1,r−1}; summing over r, **ρ*(σ') = (−1)^d t·I**.
(4) Ψ_r = action of U on V*(t_1)⊗⋯⊗V*(t_r) (independent variables, coproduct). At t_r = 0 every ρ*(e_i) on the last
factor is strictly upper triangular in one fixed flag, so Ψ_r(σ') is block triangular with diagonal blocks
Ψ_{r−1}(σ')⊗1; by induction det Ψ_r(σ')|_{(t,0,…,0)} = ±t^{d^r} ≠ 0.
(5) Faithfulness: U is a connected graded cocommutative Hopf algebra of finite type with P(U) = L; Q(U*) ≅ L*, and
ρ* injective on L means its matrix coefficients span Q(U*), hence generate U* (graded Nakayama). So each y ≠ 0 has
Ψ_r(y) ≠ 0 for some r, and Ψ_r(yσ') = Ψ_r(y)Ψ_r(σ') ≠ 0. Same on the left. k_m ≥ m − d because u_m and U agree in
degrees < m. ∎

[Verified] consistent with all data: k_m grows in every p ∤ d case computed (§4).

## 2. p | d: degeneration

* **(2,2): σ' = 0** [Proved]. e_0e_1 + e_1e_0 = [e_0,e_1] = (e_0+e_1)φ² = t ≡ 0 in L. (Checked by machine for m ≤ 6.)
  More generally, for p | d step (2) of Theorem 1 is unavailable: tr(t^kI) = 0, and L is the pgl_d-type quotient.
* (3,3): σ' = y2.2·y1.0 + y2.1·y1.2 + y2.0·y1.1 — the Coxeter term y1.0y1.1y1.2 has coefficient 3 = 0.
* **[Verified] σ' is still injective on U in low degree**: R and L injective on U_n for n ≤ 9 at (3,3), n ≤ 8 at (2,4)
  (`uinj.py`, exact, m = n + d + 1).
* **[Retracted]** An earlier remark in this session that (3,3) "strongly suggests a zero divisor" (k_3 = k_4 = k_5 = 6)
  was wrong: that degree-6 kernel is a truncation effect; on U it is 0. k_m at (3,3) must therefore jump past 6 by
  m = 10.
* Open: is σ'_{p,d} a non-zero-divisor when p | d (d ≠ p = 2)? Needs a proof not based on a T-killing representation.

## 3. The column ("odometer") mechanism is special to (2,3)  [Verified + Proved reduction]

Theorem C of the paper rests on σ' ∈ c_1U + ⋯ + c_dU (column letters c_k = e_{k−1+j}φ^k). Then every y annihilating
the column subalgebra H = ⟨c_1..c_d⟩ satisfies yσ' = 0, and the integral of u(H ∩ L_{<m}) is a kernel element in degree
d_{p,d}(m) = (p−1)(Σ_{k<m, d∤k} k + Σ_{d p^r < m} d p^r).

**Membership test** (`colideal.py`, exact in degree d, both orientations, all d columns and all d rows):

| (p,d) | dim U_δ | dim(ideal ∩ U_δ) | σ'(+1) in column ideals | σ'(−1) in row ideals |
|---|---|---|---|---|
| (2,3) | 6 | 4 | **yes** | **yes** |
| (p,2), p = 3,5,7 | 2 | 2 | yes (trivially) | yes |
| (5,3), (7,3) | 6 | 4 | no | no |
| (3,3), (2,4), (3,4) | 6, 14, 14 | 4, 8, 8 | no | no |
| (2,5), (3,5) | 30 | 16 | no | no |
| (2,7) | 126 | 64 | no | no |

The ideal has codimension 2^{d−1} − 2 in U_δ.

**Why (2,3).** For p > d there are no p-th powers in degree ≤ d, so the test is the reduction mod p of a
characteristic-0 computation (`obstruction.py`). The residual of σ'(+1) modulo the column ideal is

* d = 2: 0 (the quotient is 0-dimensional);
* d = 3: **2·(y3.0 − y2.0·y1.1)** — divisible by exactly 2;
* d = 4, 5, 6: integer vectors with gcd 1 (entries such as −3, −1, 2, 1, …).

So for d = 3 the mechanism works only in characteristic 2 (and the direct computation shows it fails at p = 3, 5, 7),
and for d ≥ 4 it fails for every p > d and for every p ≤ d tested. **The odometer structure of the paper is
a characteristic-2 accident of d = 3: it comes from the factor 2 in this obstruction.** (Classification for p ≤ d,
d ≥ 8 not done.)

Consequence: outside (2,3) and d = 2 there is no known explicit kernel family, so no a-priori quadratic upper bound
k_m ≤ d(m); the thresholds are governed by something else (e.g. (5,3): k_3 = 14 > d(3) = 12).

## 4. Thresholds and densities  [Verified, exact]

ker density x_m = dim ker R_σ'/dim u_m; the graded bound on the analogue of c_e is x_m/d. "odo" = d_{p,d}(m).

(p,d) = (3,2) (column mechanism holds, so k_m ≤ odo):

| m | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| k_m | 2 | 3 | 9 | 12 | 22 | 27 | 41 | 48 |
| odo | 2 | 6 | 12 | 12 | 22 | 34 | 48 | 48 |
| x_m/d | .278 | .204 | .097 | .076 | .050 | .042 | .032 | .028 |

(5,2):

| m | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| k_m | 4 | 6 | 18 | 24 | 42 |
| odo | 4 | 12 | 24 | 24 | 44 |
| x_m/d | .180 | .124 | .056 | .043 | .029 |

(5,3): k_m = 4, 14, 20 (m = 2, 3, 4), odo 4, 12, 24; x_m/d = .163, .071, .048.
(2,4) (p | d, orient −1): k_m = 2, 8, 12, 17 (m = 3..6), odo 3, 6, 10, 15; x_m/d = .178, .108, .085, .060.
(3,3) (p | d): k_m = 6, 6, 6 (m = 3, 4, 5; truncation, see §2); x_m/d = .126, .116, .111.
For comparison (2,3): x_m/d = .292, .193, .130, .089, .065, .052 (m = 2..7).

Observations: in every case the density decreases; for d = 2 it decreases faster than for (2,3) at equal dim u_m.
At d = 2 the threshold equals the odometer at m = 2, 5, 6, 9 (p = 3) and m = 2, 5 (p = 5) and is smaller otherwise,
so at d = 2 there are kernel elements of non-column type below the odometer. For p = 3 the odometer stalls
(odo(m) = odo(m−1)) at m = 5 and 9, and k_m = odo there; but k_6 = odo(6) as well, so "equality exactly at stalls"
is false. The gaps odo − k_m are 0, 3, 3, 0, 0, 7, 7, 0 (m = 2..9).

## 4A. The case d = 2, p odd (second round)

### 4A.1 The Lie algebra  [Proved]
Put x_k = y_{k,0}, f_k = y_{k,1} (k odd) and h_k = y_{k,0} (k even; y_{k,1} = −h_k). The bracket and p-map of §0 give
  [x_i, f_j] = 2h_{i+j},  [h_j, x_i] = x_{i+j},  [h_j, f_i] = −f_{i+j},  all other brackets of letters 0;
  x_k^{[p]} = f_k^{[p]} = 0,  h_j^{[p]} = h_{pj}.
This is sl_2 ⊗ t-loops in the principal grading (x_k = e⊗t^{(k−1)/2}, f_k = f⊗t^{(k+1)/2}, h_k = (h/2)⊗t^{k/2}), and
  σ' = x_1f_1 + f_1x_1 = 2x_1f_1 − 2h_2   (the anticommutator of the two Chevalley generators).

### 4A.2 Two explicit kernel families  [Proved; verified by direct multiplication in u_m]
For a restricted subalgebra K of L/L_{≥m} let ω_K be the product of the (p−1)-st powers of a homogeneous basis (the
integral of u(K)). By PBW, U = u(K) ⊗ V, so U/KU ≅ V (V = PBW span of the letters not in K) and ω_K·k = 0 for k ∈ K.
**Lemma.** If z ∈ u_m and zσ' ∈ K·u_m, then ω_K z σ' = 0.

(A) *Column.* H = span{x_k : k odd} ⊕ span{h_{2p^r}} is a restricted subalgebra and σ' = 2x_1f_1 − 2h_2 ∈ HU.
So ω_H ∈ ker R_σ' at every level, in degree odo(m) = (p−1)(Σ_{k<m odd} k + Σ_{2p^r<m} 2p^r).

(B) *The t²-subalgebra.* K = span{x_k (k ≡ 1 mod 4), f_k (k ≡ 3 mod 4), h_k (k ≡ 0 mod 4)} = image of sl_2[t] under
t ↦ t² (closed: [x_{4a+1}, f_{4b+3}] = 2h_{4(a+b+1)}, [h_{4a}, x_{4b+1}] = x_{4(a+b)+1}, [h_{4a}, f_{4b+3}] = −f_{4(a+b)+3},
h_{4a}^{[p]} = h_{4ap}). **Theorem.** For m ≥ 3 and 0 ≤ j < p:
  f_1^j σ' ≡ −2(2j+1)·h_2f_1^j  (mod K·u_m),
and h_2f_1^j is a basis vector of V. Hence ω_K f_1^j ∈ ker R_σ' **iff 2j + 1 ≡ 0 (mod p), i.e. j = (p−1)/2**, in degree
κ(m) = (p−1)·Σ_{k<m, k ≢ 2 (mod 4)} k + (p−1)/2.
*Proof.* f_3 ∈ K commutes with f_1, and [h_2, f_1] = −f_3, so f_1^a h_2 f_1^b ≡ f_1^{a+b}h_2 mod KU. Expanding
[x_1, f_1^j] = 2Σ_{a+b=j−1} f_1^a h_2 f_1^b gives f_1^j x_1 f_1 ≡ −2j f_1^jh_2 and f_1^{j+1}x_1 ≡ −2(j+1)f_1^jh_2, while
x_1(…) ∈ KU; finally f_1^jh_2 ≡ h_2f_1^j (the commutator is a multiple of f_3f_1^{j−1}). ∎
[Verified] (`kmech2.py`) ω_Hσ' = 0 and {j : ω_K f_1^jσ' = 0} = {(p−1)/2} exactly, for p = 3 (m ≤ 16), 5 (m ≤ 12), 7 (m ≤ 10).
At p = 2 the coefficient −2(2j+1) vanishes identically; consistent with σ' = 0 there.
By the mirror symmetry x ↔ f the same holds in the mirrored blocks. Asymptotically odo(m) ≈ (p−1)m²/4 < κ(m) ≈
(p−1)·3m²/8, so (B) matters only at small m.

### 4A.3 Thresholds: k_m = min(odo, κ) − defect  [Verified]

| p | m | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | min(odo, κ) | 2 | 3 | 9 | 12 | 22 | 27 | 41 | 48 | 66 | 66 |
| 3 | k_m | 2 | 3 | 9 | 12 | 22 | 27 | 41 | 48 | **64** | 66 |
| 5 | min(odo, κ) | 4 | 6 | 18 | 24 | 44 | 44 | | |
| 5 | k_m | 4 | 6 | 18 | 24 | **42** | 44 | | |
| 7 | min(odo, κ) | 6 | 9 | 27 | 36 | | | | |
| 7 | k_m | 6 | 9 | **26** | **34** | | | | |

All threshold kernels were printed (`kervec2.py`): at the non-defect levels they are exactly ω_H, ω_K f_1^{(p−1)/2} and
their mirrors. The defect elements (p = 5, m = 6; p = 7, m = 4, 5) are dense cancellation-type vectors (18–85 terms)
in middle Q-blocks, e.g. at p = 7, m = 5 a lowering of ω_H with h_2^6 replaced by h_2^4 plus corrections, the analogue of
η in the (2,3) paper.
**p = 3, m = 10, 11** (numba engine, `thresh_nb.py 3 2 10 48 66`, `kerblocks_nb.py`; exact): R_σ' is injective in
degrees 48..63 (and below by monotonicity), ker = 2, 4, 6 in degrees 64, 65, 66. So **k_10 = 64 = odo − 2**. The
degree-64 kernel is one vector in block (36,28) (24 terms) and its mirror (28,36): a lowered ω_H, all terms carrying
x9²x7²h6²x5², with h2² of ω_H traded for quadratic tails in x1, f1, h2, h4, f3 (e.g. the term x9²x7²h6²x5²x3²x1²f1²).
At level 11 the six blocks carrying ker^{(10)} in degrees 64, 65 ((28,36), (36,28), (28,37), (29,36), (36,29), (37,28))
have zero kernel; by the window lemma (ker^{(11)}_n ↪ ker^{(10)}_n blockwise for n < k_10 + 10 = 74) this gives
k_11 ≥ 66, and ω_H gives k_11 ≤ 66: **k_11 = 66** [Proved by computation].
**The defects die at the next level** in all cases checked: p = 3 (64 at m = 10 → none at m = 11), p = 5 (42 at m = 6 →
k_7 = 44), p = 7 (34 at m = 5 → R_σ' injective in degrees 34..62 at m = 6, so k_6 ≥ 63; predicted 66; degrees 63..66 not
computed, the run was stopped at ≈ 25 min/degree under a concurrent CPU-heavy user job; resume with
`thresh_nb.py 7 2 6 63 66`).
**[Retracted]** the bound "defect ≤ (p−1)/2" of the first version of this section: at p = 3, m = 10 the defect is 2.
Observed defects: 2 (p = 3, m = 10), 2 (p = 5, m = 6), 1 and 2 (p = 7, m = 4, 5), i.e. always ≤ 2 = deg σ'. The p = 5 defect at m = 6 dies at the next level: k_7 = 44 with a
2-dimensional kernel (`thresh.py 5 2 7 42 44`, exact; degrees < 42 by monotonicity), like η in the (2,3) paper.
[Conjecture 4A, revised] k_m ≥ min(odo(m), κ(m)) − 2 for all m (the d = 2 analogue of (H2)), and every kernel
element below min(odo, κ) dies at the next level. (Any bound k_m ≥ min(odo, κ) − o(m²), or just k_m/m^{3/2} → ∞,
suffices for the conditional theorem.)

**Numba engine** (`fam_nb.py`): same conventions as `fam.py`; monomials are base-p int64 codes, the rewriting system is
compiled with an explicit stack, blocks are built in parallel and ranked by parallel uint8 elimination mod p.
Validated (`validate_nb.py`) against all 21 exact totals of `fam.py` (both R and L where computed) and the per-degree
(5,2), m = 7 window: all equal. About 20–30× faster ((3,2), m = 9: 45 s instead of 840 s); p = 3, m = 10 needs 1–6 min
per degree (blocks up to 3.1·10^4).

### 4A.4 Density: the excess vanishes  [Verified]
With the degree baseline B_m = Σ_n max(0, h_n − h_{n+2}) (`h1check.py`):

| p = 3, m | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|
| x_m | .19342 | .15226 | .10044 | .08439 | .06335 | .05598 |
| baseline | .18519 | .14403 | .09831 | .08357 | .06323 | .05592 |
| excess | .0082 | .0082 | .0021 | .00081 | .00012 | .00006 |

p = 5: excess .0026, .0015, .00033 (m = 4, 5, 6); p = 7: .0023, .0017 (m = 4, 5).
For (2,3) the paper's excess is .0025, .0022, .0027, .0041 (m = 7..10), not decreasing. **At d = 2 right multiplication
by σ' is numerically of maximal rank up to a rapidly vanishing excess.** The baseline is O(m^{−1/2}) by the
anticoncentration lemma of the paper (shift 2 instead of 3, same proof). So graded c_e = 0 at d = 2 ⟺ excess → 0, and the
data support it much more strongly than at (2,3).
The conditional theorem of the paper transfers verbatim (duality, Hoeffding; deg σ' = 2): (H1) + Conjecture 4A ⇒
graded c_e = 0. At d = 2 the threshold is ≈ (2/3)·mid asymptotically (at (2,3): ≈ mid/2), but at the small m computed
it is within 4–6 of mid, so the (H1) ratio near mid is not yet informative (2–4 for most m, 11–14 at p = 3, m = 8, 9
where h_{n−k_m} is tiny).

## 5. Retractions relative to the earlier plan

* "d = 2 may give an unconditional c_e = 0 because σ' is a Cartan letter with kernel density 2^{−R}": **wrong**. At
  p = 2 σ' = 0; at p odd σ' = 2e_0e_1 − [e_0,e_1] is not a letter, and its kernel density (table above) is a genuine
  problem of the same type as (2,3).
* "These arguments hold verbatim for all (p, d)": true for the non-zero-divisor theorem when p ∤ d; false for the
  odometer kernel (only (2,3) and d = 2) and false for the representation-theoretic proof when p | d.
* "Defect pattern d | m, m ≠ d·p^r": moot outside (2,3), since there is no ω-family to have defects.

## 6. Open questions / next steps

1. p | d: prove or refute that σ' is a non-zero-divisor (data: injective in low degree at (3,3), (2,4)).
2. d = 2, p odd (see 4A): prove Conjecture 4A (bounded defect) or at least a superlinear lower bound for k_m; prove
   excess → 0 directly (maximal rank of σ' = {x_1, f_1} on u_m up to o(|G_m|)). Understand the defect elements.
3. (5,3), (2,5): what produces the first kernel? (No column/row ideal contains σ'.)
4. A numba engine for general p (the present Python engine stops near dim u_m ≈ 5·10^5).
5. Group level: construct the LSV lattice and the Hecke element for (q,d) = (3,2), (2,4), (5,2) and check that the
   leading form is the cyclic Coxeter sum (sign conventions: the mod-p Laplacian is −I + A, so F = 1 − f̄⋯f̄).
