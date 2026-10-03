# THEORY — split form of gr P, the leading form sigma(F), and the kernel of R_sigma

Labels used throughout:
**[Proved]** = complete proof written here (modulo standard facts cited by name);
**[Proved + computation]** = proof whose only non-conceptual input is a finite exact computation, with the script named;
**[Verified]** = exact computer verification for the stated finite range only (no proof);
**[Conjecture]** / **[Heuristic]** = not proved.

Scripts referred to are in `ce_compute/`; numerical results are in `RESULTS.md`, section "Task C".

---------------------------------------------------------------------------------------------------
## 0. Notation

* F8 = F2(alpha), alpha^3 = alpha + 1, sigma(a) = a^2. A := F8[phi; sigma] (skew polynomials, phi·a = sigma(a)·phi),
  graded by deg phi = 1. Z(A) = F2[t], t = phi^3. (sigma has order 3, so phi^k is central iff 3 | k; the fixed field is F2.)
* L = gr P with L_k = F8 phi^k (3 ∤ k), F8 phi^k / F2 t^{k/3} (3 | k), [a phi^i, b phi^j] = (a sigma^i(b) + b sigma^j(a)) phi^{i+j},
  (a phi^k)^{[2]} = a sigma^k(a) phi^{2k}. d_k = dim L_k = 3 (3 ∤ k), 2 (3 | k).
* u(L) = restricted enveloping algebra; u_m := u(L / L_{>=m}) (= gr F2[G_m] by Jennings, see RESULTS Task 2/5);
  N_m := dim L/L_{>=m} = 3(m-1) - floor((m-1)/3), |G_m| = 2^{N_m} = dim u_m.
* sigma F = x1.1 x1.α x1.α² + x1.1 x2.α + x1.α² x2.α + x1.1 x2.α² + x1.α x2.α² ∈ u(L)_3 (RESULTS Task 2), R_sigma: y -> y·sigma F.
* k_m := least n with R_sigma : (u_m)_n -> (u_m)_{n+3} non-injective.  S_m := m(m-1)/2.

---------------------------------------------------------------------------------------------------
## 1. The split form (Task C1)

### 1.1 L as a quotient of an associative algebra  [Proved]
For any associative graded algebra B, B_{>0} with [x,y] = xy - yx and x^{[2]} = x^2 is a graded restricted Lie algebra.
For B = A: aphi^i·bphi^j - bphi^j·aphi^i = (a sigma^i(b) + b sigma^j(a)) phi^{i+j} and (a phi^k)^2 = a sigma^k(a) phi^{2k}.
The central subspace T := t F2[t] ⊂ A_{>0} is closed under squaring, hence a restricted ideal, and
**L ≅ A_{>0} / T** as graded restricted Lie algebras (degree k: F8 phi^k modulo F2 t^{k/3} when 3 | k).

### 1.2 Splitting over F8  [Proved]
Let E = F2^3 (componentwise product, idempotents e_0, e_1, e_2), s(e_p) = e_{p+1} (indices mod 3), A^s := E[phi; s]
(phi·e_p = e_{p+1}·phi), T^s := t F2[t] with t = (1,1,1) phi^3 (central), and **L^s := A^s_{>0} / T^s**.
Define
        psi : A ⊗_{F2} F8 -> A^s ⊗_{F2} F8,    a phi^k ⊗ lambda  |->  (lambda a, lambda sigma^{-1}(a), lambda sigma^{-2}(a)) phi^k .
*Proof that psi is an isomorphism of graded F8-algebras with psi(T ⊗ F8) = T^s ⊗ F8.* In degree 0, psi_0 : F8 ⊗ F8 -> F8^3,
a ⊗ lambda |-> (lambda sigma^{-p}(a))_p is the classical isomorphism F8 ⊗_{F2} F8 ≅ prod_{Gal(F8/F2)} F8 (a ring isomorphism).
It intertwines sigma ⊗ 1 with the shift: psi_0(sigma(a) ⊗ lambda)_p = lambda sigma^{1-p}(a) = psi_0(a ⊗ lambda)_{p-1} = s(psi_0(a⊗lambda))_p.
Hence psi_0 extends to the skew polynomial rings (phi |-> phi), bijectively in each degree, and t = phi^3 |-> phi^3 = (1,1,1)phi^3. ∎
Consequently (restricted structures are defined by the products):
**L ⊗ F8 ≅ L^s ⊗ F8,   u(L) ⊗ F8 ≅ u(L^s) ⊗ F8,   u_m ⊗ F8 ≅ u^s_m ⊗ F8**   (u^s_m := u(L^s / L^s_{>=m})),
all graded, the last because psi preserves degree, so L_{>=m} ⊗ F8 corresponds to L^s_{>=m} ⊗ F8.
Basis letters of L^s: y_{k,p} := e_p phi^k (p = 0,1,2 if 3 ∤ k; p = 0,1 if 3 | k, with e_2 phi^k ≡ y_{k,0} + y_{k,1} mod T^s).
Brackets: e_p phi^i · e_q phi^j = [q ≡ p - i] e_p phi^{i+j}, so
[y_{i,p}, y_{j,q}] = [q ≡ p-i] e_p phi^{i+j} + [p ≡ q-j] e_q phi^{i+j},   y_{k,p}^{[2]} = [3 | k] e_p phi^{2k}.
Image of the F8-letters: psi(b phi^k) = sum_p sigma^{-p}(b) y_{k,p} (e.g. x1.α -> α y1.0 + α^4 y1.1 + α^2 y1.2).
*Machine check* (`taskC1_iso.py`, `out_taskC1_iso.txt`): all 351 brackets and 27 squares of basis letters of degree <= 10 are
preserved, psi is bijective in each degree, and psi(YZ) = psi(Y)psi(Z) on 88 random products in u (independent
straightening codes for the two models).
*Remark (F2-forms).* L^s and L are different F2-forms of the same F8-Lie algebra: L = fixed points of rho^{-1} ∘ (1 ⊗ Frob)
on L^s ⊗ F8, where rho(e_p phi^k) = e_{p+1} phi^k is the vertex rotation (§1.4). L^s has u(L^s) full of zero divisors
already in degree (1,1) (RESULTS 3c), u(L) does not; only after ⊗F8 do they agree.

### 1.3 Identification with the positive part of affine A_2  [Proved]
Put Pi := E_{10} + E_{21} + t E_{02} ∈ M_3(F2[t]) (E_{ij} matrix units, indices mod 3). Then Pi E_{qq} = E_{q+1,q+1} Pi and Pi^3 = t·I,
so  sum_k D_k phi^k |-> sum_k D_k Pi^k  (D_k diagonal) is an injective graded algebra map A^s -> M_3(F2[t]) whose image is the
Iwahori order {X : X(0) lower triangular}; A^s_{>0} maps onto its radical {X : X(0) strictly lower triangular}, graded by
deg E_{ij} t^n = 3n + (i - j) (the **principal grading**); e_p phi |-> E_{p,p-1} (p = 1, 2), e_0 phi |-> t E_{02}.
Since 3 is invertible in F2, gl_3 = sl_3 ⊕ F2·I, and x |-> x + tr(x)·I identifies gl_3/F2 I with sl_3 as restricted Lie algebras
(tr(x^2) = tr(x)^2 in characteristic 2). Hence
**L^s ≅ (strictly-lower-mod-t part of sl_3(F2[t])) = n̂ ,** the nilradical of the standard Borel of the loop algebra
sl_3 ⊗ F2[t, t^{-1}] in the principal grading, i.e. the positive part of affine A_2^{(1)} (the central extension and
derivation of the Kac–Moody algebra do not meet the positive part; transpose-inverse turns "lower" into the usual n̂_+).
The degree-1 letters are the three Chevalley generators; the 2-map is the matrix square: real root vectors E_{ij} t^n
(i ≠ j) have square 0, Cartan elements h t^n square to h^2 t^{2n} (= h t^{2n} over F2 for 0/1-diagonal h). Root
multiplicities: d_k = 3 (three real roots of principal height k, 3 ∤ k) and d_k = 2 (imaginary root (k/3)δ, multiplicity
rank sl_3 = 2). **The hypothesis of the task is correct**, with the precise F2-forms of §1.2.

### 1.4 The Q-grading and the symmetries  [Proved]
A^s is the path algebra of the oriented 3-cycle (vertices Z/3, arrow e_p phi from p-1 to p). Grading by arrow multiplicities:
Q-weight of e_p phi^k := sum_{s=0}^{k-1} beta_{(p - s) mod 3} ∈ Q_+ = N beta_0 ⊕ N beta_1 ⊕ N beta_2 (beta_p := weight of e_p phi);
products of paths add weights, t^j has weight j·delta, delta := beta_0 + beta_1 + beta_2, so L^s, u(L^s) and u^s_m are
Q-graded, refining the principal degree n = n_0 + n_1 + n_2 (beta_p = simple roots, delta = null root).
* rho : e_p phi^k |-> e_{p+1} phi^k is an algebra automorphism of A^s fixing T^s; hence an automorphism of L^s, u^s_m, acting on
  weights by (n_0,n_1,n_2) |-> (n_2,n_0,n_1).
* tau : e_p phi^k |-> e_{k-p} phi^k is an algebra ANTI-automorphism of A^s fixing T^s (check: e_p phi^i · e_q phi^j =
  [q ≡ p-i] e_p phi^{i+j} |-> [q ≡ p-i] e_{i+j-p} phi^{i+j}, while tau(e_q phi^j)·tau(e_p phi^i) = e_{j-q} phi^j · e_{i-p} phi^i =
  [i-p ≡ j-q-j] e_{j-q} phi^{i+j}, and j - q = i + j - p when q ≡ p - i). In characteristic 2 an anti-automorphism of A^s is a restricted Lie automorphism of L^s
  ([tau x, tau y] = tau(yx) - tau(xy) = tau[x,y]); Theta' := S ∘ u(tau) (S = antipode) is an anti-automorphism of u^s_m with
  Theta'(z_1 ⋯ z_r) = tau(z_r) ⋯ tau(z_1), acting on weights by theta(n_0,n_1,n_2) = (n_1,n_0,n_2).

### 1.5 sigma F in the split basis  [Proved + computation: `taskC1_iso.py`, `taskC3_lemmas.py`]
        **sigma' := psi(sigma F) = y1.0·y1.1·y1.2 + y1.2·y2.1 + y1.0·y2.2 + y3.1**        (products in u; terms with a letter of
degree >= m are dropped in u_m). Equivalent forms (checked for m = 3..14):
        sigma' = y1.2·y1.1·y1.0 + y1.1·y2.0 + y3.0
               = c_1·(y1.1·y1.2 + y2.2) + c_2·y1.2 + c_3,    c_k := e_{k-1} phi^k  (c_1 = y1.0, c_2 = y2.1, c_3 = e_2 phi^3 = y3.0 + y3.1).
Proof of the equivalences by hand, using [y1.1, y1.0] = y2.1, [y1.2, y1.1] = y2.2, [y1.2, y1.0] = y2.0 (as Lie elements
[e_2 phi, e_0 phi] = e_0 phi^2 etc.), [y2.0, y1.1] = y3.0 + y3.1, [y1.2, y2.1] = y3.0: e.g. y1.2·y2.1 = y2.1·y1.2 + y3.0, so
sigma' = y1.0(y1.1 y1.2 + y2.2) + y2.1 y1.2 + (y3.0 + y3.1). The identity psi(sigma F) = sigma' itself is a finite exact
computation (expanding the images of the six letters of degree 1, 2).
Facts: **(a)** sigma' has coefficients in F2; **(b)** sigma' is Q-homogeneous of weight delta (each of the four terms
has weight (1,1,1)); **(c)** rho(sigma') = sigma' (hand check: rho(y1.0 y1.1 y1.2) = y1.0y1.1y1.2 + y1.2y2.1 + y1.1y2.0 + y3.0,
rho(y1.2y2.1) = y1.0y2.2, rho(y1.0y2.2) = y1.1y2.0, rho(y3.1) = y3.0 + y3.1; the sum is sigma'); **(d)** Theta'(sigma') = sigma'.
(The F8-model letters are not Q-homogeneous; the homogeneity of sigma' is a genuine property of F = 1 + fbar_1 fbar_0 fbar_2.)

### 1.6 Consequences for ranks  [Proved]
(i) For every m and n: rank_{F2}(R_{sigma F} | (u_m)_n) = rank_{F8}(R ⊗ F8) = rank_{F8}(R_{sigma'} ⊗ F8 | (u^s_m ⊗ F8)_n)
= rank_{F2}(R_{sigma'} | (u^s_m)_n)   (psi is an F8-algebra isomorphism carrying sigma F to sigma', and sigma' is defined over F2).
So all ranks, kernels dimensions, cokernel dimensions, Jordan types of R_sigma on u_m can be computed in u^s_m over F2.
(ii) By (b), R_{sigma'} = ⊕_gamma R_gamma, R_gamma : (u^s_m)_gamma -> (u^s_m)_{gamma+delta}.
(iii) By (c), rank R_gamma = rank R_{rho(gamma)}.
(iv) **Duality.** rank R_gamma = rank R_{theta(Gamma - gamma - delta)}, Gamma := weight of the top degree of u^s_m.
*Proof sketch:* u^s_m is a finite-dimensional graded connected Hopf algebra with one-dimensional top degree; the top
coefficient lambda is a Frobenius form, so (a,b) |-> lambda(ab) pairs (u^s_m)_gamma and (u^s_m)_{Gamma-gamma} perfectly, and
R_{sigma'} on block gamma is adjoint to L_{sigma'} on block Gamma - gamma - delta; Theta' conjugates L_{sigma'} on block mu to
R_{sigma'} on block theta(mu) because Theta'(sigma') = sigma'. [Verified on all blocks for m = 3..10, `taskC4_fits.py`.]
In principal degrees: dim ker(R | u_n) = dim coker(R | u_{top-n}), which explains the symmetry of all kernel tables.

---------------------------------------------------------------------------------------------------
## 2. Column subalgebras and kernel elements (Task C2)

### 2.1 Column subalgebras  [Proved]
For j ∈ Z/3 let c^{(j)}_k := e_{k-1+j} phi^k (k >= 1) and C^{(j)} := span{c^{(j)}_k} ⊂ L^s (mod T^s). In matrix terms (§1.3) these
are the root vectors whose matrices are supported in one fixed column (a "column subalgebra"). In A^s:
c_a c_b = [3 | b] c_{a+b}, hence
        [c_a, c_b] = ([3|a] + [3|b]) c_{a+b},     c_a^{[2]} = [3|a] c_{2a}.
So C^{(j)} is a restricted subalgebra, the c_k with 3 ∤ k span an abelian ideal with zero 2-map, and the c_{3i} act on it
by c_k -> c_{k+3i}; rho(C^{(j)}) = C^{(j+1)}.

### 2.2 The subalgebra generated by c_1, c_2, c_3  [Proved]
H^{(j)} := restricted subalgebra generated by c_1, c_2, c_3 equals span{c_k : 3 ∤ k} ⊕ span{c_{3·2^r} : r >= 0}.
*Proof.* The right side is closed: [c_a, c_b] ≠ 0 forces exactly one of a, b to be divisible by 3, so a+b is not; and
c_a^{[2]} ≠ 0 forces 3 | a = 3·2^r, giving c_{3·2^{r+1}}. It is generated: c_{k+3} = [c_k, c_3] (3 ∤ k) gives all c_k, 3 ∤ k,
from c_1, c_2, and c_{3·2^{r+1}} = c_{3·2^r}^{[2]}. ∎  In L^s/L^s_{>=m}: H_{<m} = span{c_k : k ∈ 𝓗_m},
        𝓗_m := {1 <= k < m : 3 ∤ k} ∪ {3·2^r < m},     **d(m) := sum_{k ∈ 𝓗_m} k.**
d(m) for m = 2..16: 1, 3, 6, 10, 15, 21, 28, 36, 36, 46, 57, 69, 82, 96, 96 — equal to S_m = m(m-1)/2 exactly for m <= 9,
and d(m) = m^2/3 + O(m) in general (multiples of 3 other than 3·2^r are missing: 9, 15, 18, 21, 27, ...).
**The "binary odometer" heuristic of the task is the 2-map chain c_3 -> c_6 -> c_12 -> ...: the squares of the imaginary
column root vectors carry along the powers of 2, while all real root vectors square to 0; c_9, c_15, c_18, ... are never
reached by carries, and this is exactly why the threshold departs from m(m-1)/2 from m = 10 on.**

### 2.3 Integrals and annihilators  [Proved]
Let K ⊂ L^s/L^s_{>=m} be a graded restricted subalgebra with homogeneous basis z_1,…,z_r (positive degrees), and
omega_K := z_1 z_2 ⋯ z_r ∈ u^s_m. Then:
(a) omega_K spans the top degree D = sum deg z_i of u(K) and does not depend (up to the scalar 1) on the order of the factors;
(b) **{y ∈ u^s_m : y·z = 0 for all z ∈ K} = u^s_m · omega_K**, a space of dimension 2^{N_m - r}, with Hilbert series
    z^{D} · prod_{k<m} (1+z^k)^{d_k} / prod_i (1 + z^{deg z_i}).
*Proof.* PBW for restricted enveloping algebras (any ordered basis): with a complementary homogeneous basis w_1..w_s of L^s/L^s_{>=m}
ordered before the z's, multiplication V ⊗ u(K) -> u^s_m (V = span of the monomials in the w's) is a linear isomorphism and
an isomorphism of right u(K)-modules. So y = sum_a w^a f_a annihilates K iff each f_a does, i.e. f_a·u(K)^+ = 0, i.e. f_a is a
right integral of the finite-dimensional Hopf algebra u(K); these form a one-dimensional space (Larson–Sweedler), and
omega_K is one (omega_K·u(K)^+ lies above the top degree). Any ordered product of the z's is a PBW monomial for that order,
hence a nonzero element of the 1-dimensional top degree. ∎

### 2.4 Theorem (explicit kernel)  [Proved]
For every m >= 2 and j ∈ Z/3, with omega_j := omega_{H^{(j)}_{<m}} = prod_{k ∈ 𝓗_m} c^{(j)}_k:
        **u^s_m · omega_0 + u^s_m · omega_1 + u^s_m · omega_2  ⊂  ker R_{sigma'}.**
The three omega_j are nonzero of degree d(m) and have pairwise different Q-weights, hence are linearly independent. In
particular **k_m <= d(m)** and dim ker(R_sigma | (u_m)_{d(m)}) >= 3, for the original sigma F on u_m (via §1.6).
*Proof.* By §1.5, sigma' ∈ c_1 u + c_2 u + c_3 u (for j = 0; for j = 1, 2 apply rho, using rho(sigma') = sigma'). If y·c_k = 0
for c_1, c_2, c_3 then y sigma' = 0. By 2.2–2.3, {y : y c_1 = y c_2 = y c_3 = 0} = {y : y H_{<m} = 0} = u^s_m omega_j (a vector
annihilating generators annihilates the restricted subalgebra they generate, since y(ab) = (ya)b, y(a^{[2]}) = (ya)a).
Weights: wt(c^{(0)}_k) = (n_0,n_1,n_2) with n_r = #{0 <= s < k : s ≡ r}, so n_0 - n_2 = 1 for 3 ∤ k and 0 for 3 | k; hence
wt(omega_0) has n_0 - n_2 = #{k ∈ 𝓗_m : 3 ∤ k} >= 1: not rotation invariant, so the three rotations differ. ∎
[Machine checks of omega_H ≠ 0, omega_H·H = 0, omega_H·sigma' = 0 for m = 3..14: `taskC3_lemmas.py`.]

Translated back to the F8 model this is the "one letter of each degree" structure observed in Task A2: for m <= 9,
H_{<m} has exactly one letter in each degree 1..m-1 and omega_j = c_1 c_2 ⋯ c_{m-1}; its psi^{-1}-image is a sum of PBW
monomials with one letter of each degree (the 91–93, 183–184 terms of `out_taskA2_kernel.txt`).

### 2.5 The kernel at the threshold, m <= 11  [Verified]  (`taskC1_blocks.py`, `taskC2_low.py`, `taskC2_kernel.py`, `taskC_check_block.py`)
* m = 2..8, 10, 11: **k_m = d(m)** and ker(R | (u_m)_{d(m)}) = span{omega_0, omega_1, omega_2} exactly (3-dimensional, one
  dimension in each of the three rotated Q-blocks, each spanned by the single split PBW monomial omega_j).
  Values: k_m = 1, 3, 6, 10, 15, 21, 28 (= S_m) for m = 2..8; **k_10 = 36 (S_10 = 45), k_11 = 46 (S_11 = 55)**.
  E.g. m = 10: omega = y8.0·y7.2·y6.1·y5.0·y4.2·y3.1·y2.0·y1.2 (j = 2; no letter of degree 9), m = 11:
  omega = y10.1·y8.2·y7.1·y6.0·y5.2·y4.1·y3.0·y2.2·y1.1 (j = 1; no degree 9).
* **m = 9 is exceptional: k_9 = 35 = d(9) - 1**: the kernel in degree 35 is 3-dimensional, spanned by eta_j = rho^j(eta),
        eta = y8.0·y7.2·y6.1·y5.0·y4.2·y3.1·y2.0 + y8.0·y7.2·y6.1·y5.0·y4.0·y4.2·y1.2 + y7.0·y7.2·y6.1·y5.0·y4.2·y3.1·y2.0·y1.2
            + y8.0·y6.0·y6.1·y5.0·y4.2·y3.1·y2.0·y1.2 + y8.0·y7.2·y5.0·y5.1·y4.2·y3.1·y2.0·y1.2 + y8.0·y7.2·y6.1·y4.0·y4.2·y3.1·y2.0·y1.2
            + y8.0·y7.2·y6.1·y5.0·y3.0·y3.1·y2.0·y1.2 + y8.0·y7.2·y6.1·y5.0·y4.2·y2.0·y2.1·y1.2 + y8.0·y7.2·y6.1·y5.0·y4.2·y3.1·y1.0·y1.2
  (descending-degree PBW order; weight wt(omega) - beta_2, omega = y8.0·y7.2·y6.1·y5.0·y4.2·y3.1·y2.0·y1.2 = omega_{j=2}).
  Structurally eta is a "lowered staircase": its terms are omega with one factor c_k = e_{k+1}phi^k replaced by
  e_{k+1}phi^{k-1} (= ad(e_1 phi^{-1})(c_k) up to a Cartan term), i.e. eta ≈ ad(f)(omega) for the negative Chevalley
  generator f of weight -beta_2. eta is not annihilated by any single letter (`taskC2_eta_probe.py`, `out_taskC2_eta_probe.txt`), so it is
  not of the form u·omega_K; it arises from a cancellation inside eta·sigma'. Independently confirmed with the other
  straightening code (`taskC_check_block.py`, `out_taskC_check_m9.txt`). At m = 10 it is no longer in the kernel
  (k_10 = 36).
* Consequently the empirical rule "R_sigma injective exactly below S_m = m(m-1)/2, kernel 3-dimensional there" holds for
  m = 2..8 and **fails for m = 9 (k_9 = 35 < 36) and m >= 10 (k_m <= d(m) < S_m)**.

---------------------------------------------------------------------------------------------------
## 3. What can be proved about injectivity (Task C3)

### 3.1 Central filtration  [Proved]
L_m is central in L/L_{>=m+1} ([L_i, L_m] ⊂ L_{>=m+1}) with zero 2-map (L_m^{[2]} ⊂ L_{2m}), so u(L_m) = Λ(L_m) (exterior
algebra, dim 2^{d_m}), and by PBW u_{m+1} is a free Λ(L_m)-module with u_{m+1}/L_m u_{m+1} = u_m. Let J = L_m u_{m+1}
(a two-sided ideal, J^i/J^{i+1} ≅ Λ^i(L_m) ⊗ u_m). R_sigma preserves the J-adic filtration and gr R_sigma = 1 ⊗ R_sigma^{(m)}.
Since dim ker of a filtered map <= dim ker of its associated graded map, for every principal degree n and (split model)
every Q-block gamma:
        dim ker R^{(m+1)}_gamma  <=  sum_{S ⊂ basis of L^s_m}  dim ker R^{(m)}_{gamma - wt(S)}.
Corollaries:
(a) **k_{m+1} >= k_m** (the first non-injective degree is nondecreasing in m).
(b) **dim ker R^{(m+1)} / |G_{m+1}| <= dim ker R^{(m)} / |G_m|**; equivalently the graded upper bound
    (graded coker)/n_m for c_e is nonincreasing in m. (Total kernel = total cokernel for an endomorphism.)
(c) With (a) and the verified k_11 = 46: **R_sigma is injective on (u_m)_n for all m >= 11 and all n <= 45** [Proved + computation].
(d) The inequality is the certification tool used in `taskC2_low.py` (a block is certified injective if all the
    blocks on the right are known to be injective).

### 3.2 Lower bound  [Proved, trivial]
dim ker R_gamma >= max(0, dim u_gamma - dim u_{gamma+delta}). Summing over blocks,
        dim ker R_sigma  >=  B^Q_m := sum_gamma max(0, dim u_gamma - dim u_{gamma+delta})  >=  B_m := sum_n max(0, h_n - h_{n+3})
(the middle inequality because sum of positive parts >= positive part of the sum, degree by degree). B^Q_m and B_m agree
to 4 digits for m <= 15 (RESULTS). **No element of Q-weight delta (in particular sigma') can give a graded cokernel
below B^Q_m, and no element of degree 3 at all can go below B_m** (for degree-3 elements the blockwise bound need not apply).
B_m ≈ 2.55 m^{-3/2} |G_m| (§4).

### 3.3 Mechanism of the threshold and of its jumps  [Proved, partial]
Theorem 2.4 gives k_m <= d(m). The inductive step m -> m+1 of 3.1 shows why omega^{(m)} survives exactly when m ∉ 𝓗_{m+1}:
if m is a multiple of 3 not of the form 3·2^r (m = 9, 15, 18, 21, ...), then H_{<m+1} = H_{<m}, omega^{(m+1)} = omega^{(m)} and
the kernel element persists in the same degree (k_10 <= d(10) = d(9) = 36) [proved]; otherwise H_{<m+1} gains c_m, and
omega^{(m)}·c_k (k ∈ 𝓗_m) can now carry into the new central letter c_m, so omega^{(m)} need not stay in the kernel; in all
verified cases it leaves the kernel and the threshold moves up by m [observed, m <= 11].
**What is NOT proved:** injectivity below d(m) (statement (i) of the task with S_m replaced by d(m)) for general m, and
exactness of the kernel at d(m) (statement (ii)). The natural proof route — show by induction on m via 3.1 that the
connecting map ker R^{(m)}_n -> coker(R on J)_{n+3} is injective for n < d(m+1) — requires control of coker R^{(m)} in
degrees n+3-m, which near d(m) lie in the region where R^{(m)} is far from surjective; the m = 9 element eta shows that
there are non-annihilator ("cancellation") kernel elements, so any proof must be finer than a PBW leading-term argument.
A leading-term (Gröbner) proof in the split basis is also obstructed by the fact that the leading monomial of sigma' in
any monomial order is y1.0 y1.1 y1.2 (or a degree-2·1 product), whose right multiplication is itself highly non-injective
(the split u has zero divisors in degree (1,1)).

### 3.4 Conjectures
* [Conjecture] k_m ∈ {d(m) - 1, d(m)} for all m >= 2, with k_m = d(m) and kernel span{omega_j} for all but finitely many m.
  (Verified m <= 11; m = 12 see RESULTS.)  In particular k_m ~ m^2/3, not m^2/2.
* [Conjecture] The lower-half kernel (degrees below the middle) is generated, as a left u^s_m-module, by the omega_j and
  a few sporadic elements like eta; it is strictly larger than sum_j u omega_j (Hilbert series comparison in RESULTS C4).

---------------------------------------------------------------------------------------------------
## 4. Density (Task C4)

* [Proved] x_m := dim ker R_sigma / |G_m| (= graded coker / |G_m| = 3 × graded bound for dim C_m / n_m) is nonincreasing (3.1b).
* [Proved] x_m >= B^Q_m / |G_m|, and B_m/|G_m| = 3 h_max/|G_m| (1 + o(1)) ~ 3/(sqrt(2 pi) sigma_m) with sigma_m^2 = sum_{k<m} k^2 d_k / 4
  ~ 2m^3/9 (local CLT for the degree of a random subset of letters; sketch) — numerically B_m/|G_m| = 2.55..2.75·m^{-3/2}
  for m <= 90 (`taskC4_baseline.py`). So the maximal-rank part tends to 0 like m^{-3/2}.
* [Verified] x_m = B^Q_m/|G_m| + e_m with excess e_m = 0.0469, 0.0117, 0.0146, 0.0095, 0.0025, 0.0022, 0.0027, 0.0041
  (m = 3..10): the kernel is the maximal-rank kernel plus 2–5% (m = 8..10). The excess is symmetric about the middle
  degree (3.1.6(iv)); below the middle it equals the whole kernel, which starts at k_m ≈ m^2/3.
* [Heuristic] c_e = 0 iff e_m -> 0. Counting suggests the lower-half kernel at the middle degree is at most of the order of
  dim u(L)_{mid - k_m} ≈ exp(pi sqrt((8/3)(m^2/3)/3)) = e^{1.71 m}, against h_mid(u_m) ≈ 2^{8m/3}/m^{3/2} = e^{1.85 m}/m^{3/2};
  this would give e_m -> 0 but only very slowly (the ratio m^{3/2} e^{-0.14 m} still grows up to m ≈ 11). The data
  (e_m rising from 0.0022 to 0.0041 between m = 8 and 10) are consistent with that and do not decide the question.

---------------------------------------------------------------------------------------------------
## 5. The excess (Task D)

Notation: d_γ := dim (u^s_m)_γ, rank_γ := rank R_γ, base_γ := max(0, d_γ − d_{γ+δ}),
**excess_γ := min(d_γ, d_{γ+δ}) − rank_γ = ker_γ − base_γ ≥ 0**, e_m := Σ_γ excess_γ / |G_m|, mid := (top − 3)/2,
γ* := θ(Γ − γ − δ) (§1.6(iv)), X_m := degree of a uniformly random PBW monomial of u_m (so P(X_m = n) = h_n/|G_m|).
Numerical results: RESULTS, "Task D".

### 5.1 Symmetry of the excess and reduction to the lower half  [Proved, given §1.6(iv)]
excess_γ = excess_{γ*}, and |γ*| = top − 3 − |γ|. Consequently
        e_m·|G_m| = 2·Σ_{|γ| < mid} excess_γ + Σ_{|γ| = mid} excess_γ  ≤  2·Σ_{n ≤ mid} dim ker(R_σ' | (u_m)_n).
*Proof.* θ fixes δ and preserves dimensions, and the Frobenius pairing gives d_μ = d_{Γ−μ}; hence d_{γ*} = d_{Γ−γ−δ} = d_{γ+δ}
and d_{γ*+δ} = d_{θ(Γ−γ)} = d_γ. With rank_{γ*} = rank_γ (§1.6(iv)): excess_{γ*} = min(d_{γ+δ}, d_γ) − rank_γ = excess_γ.
The inequality uses excess_γ ≤ ker_γ. ∎

### 5.2 Annihilator-type kernels are exactly the column kernels  [Proved + computation: `taskD2_ideals.py`, `taskD2_ann.py`]
Let m ≥ 4 and let K ⊂ L^s/L^s_{≥m} be a Q-graded restricted subalgebra with σ' ∈ K·u^s_m. Then K ⊇ H^{(j)}_{<m} for some j,
hence Ann_right(K) = u·ω_K ⊂ u·ω_j. So Σ_{K : σ' ∈ K u} Ann(K) = A := u ω_0 + u ω_1 + u ω_2 (by §2.4 A ⊂ ker R_σ').
*Proof.* Let z_i be a homogeneous basis of K. K·u = Σ z_i u, so taking the weight-δ component, σ' ∈ Σ_{wt z_i ≤ δ} z_i·u_{δ − wt z_i}.
The Lie elements of weight ≤ δ are spanned by y1.p (weight β_p), y2.p (β_p + β_{p−1}) and span{y3.0, y3.1} (weight δ) (no
letter has weight 2β_p or β_p + β_q other than these), so K_{≤δ} := K ∩ (weight ≤ δ) is spanned by a subset of the six letters
y1.p, y2.p together with a subspace of span{y3.0, y3.1}, and σ' ∈ K_{≤δ}·u. All these finitely many subspaces were enumerated
(`taskD2_ideals.py`): σ' ∈ K_{≤δ}·u holds exactly for supersets of 19 minimal ones; for each of them the restricted subalgebra
they generate contains c^{(j)}_1, c^{(j)}_2, c^{(j)}_3 for some j (`taskD2_ann.py`, checked for m = 5..10). Both checks only
involve degrees ≤ 3, where u^s_m does not depend on m ≥ 4. Then H^{(j)} ⊂ K, and Ann(K) ⊂ Ann(H^{(j)}) = u ω_j by §2.3. ∎
**Size:** dim u ω_j = 2^{N_m − |𝓗_m|}, so dim A ≤ 3·2^{N_m − |𝓗_m|}, and |𝓗_m| ≥ #{k < m : 3 ∤ k} ≥ 2(m−1)/3:
        **dim A / |G_m| ≤ 3·2^{−2(m−1)/3} → 0.**

### 5.3 Structural lower bound and the part of the excess it explains  [Proved]
With a_γ := dim A_γ:  ker_γ ≥ S_γ := max(a_γ, d_γ − d_{γ+δ} + a_{γ*}), and Σ_γ (S_γ − base_γ) ≤ 2 dim A ≤ 6·2^{N_m − |𝓗_m|}.
*Proof.* rank_γ ≤ d_γ − a_γ since A_γ ⊂ ker R_γ; rank_γ = rank_{γ*} ≤ d_{γ*} − a_{γ*} = d_{γ+δ} − a_{γ*}. Finally
max(a, d − d' + a*) − max(0, d − d') ≤ a + a*. ∎
So the excess "forced by the ω_j" has density ≤ 6·2^{−2(m−1)/3} → 0. [Verified, `taskD2_ann.py`] It accounts for 80, 69, 67, 41,
17, 25 % of the excess for m = 5..10; the rest (ker_γ − S_γ > 0 on 3, 36, 42, 187, 596, 972 blocks) comes from kernel elements of
cancellation type (η at m = 9; at m = 10 single kernel vectors in the blocks wt(ω_j) + 2β_{j'} in degree 38, then 3·(1,1,2,2,4,4,5,6)
new left-module generators in degrees 38, 40, …, 50; `taskD2_module.py`).

### 5.4 Generic elements of weight δ  [Verified]
(u^s_m)_δ is 6-dimensional (basis y3.0, y3.1, y2.q·y1.{q+1}, y1.0·y1.1·y1.2). By lower semicontinuity of rank, the generic element
x = Σ t_i b_i (t_i indeterminates) has rank_γ(x) ≥ rank_γ(x(c)) for every specialization c over any extension of F2.
* m = 4..8: on every Q-block some specialization (an F2-element, or a random F16-element) has maximal rank min(d_γ, d_{γ+δ}):
  **the generic element of weight δ has maximal rank on every block (excess 0)** (`taskD1_maxrank.py`).
* m = 9: the same holds on all blocks but 18 square blocks (ρ/duality orbits of (12,17,18), (12,18,17), (14,15,20) of size 15
  and (13,16,19), (13,19,16) of size 43), where all specializations tried (63 over F2, 8 over F16/F256, 6 over F_{2^16}) have
  corank 1. Hence generic excess ≤ 18 [Verified], and = 18 with Schwartz–Zippel error < 10^{-18} [randomized]. There is no common
  annihilator of (u_9)_δ in these blocks. m = 10 (blocks of dim ≤ 3000): generic excess ≤ 846, versus 12060 for σ'.
So maximal rank ("Lefschetz property for weight δ") holds generically for m ≤ 8 and fails only in thin, strongly unbalanced
blocks from m = 9 on. The excess of σ' (11200 at m = 9) is a special-position phenomenon. The F2-element
τ = y2.2·y1.0 + y2.0·y1.1 + y2.1·y1.2 (ρ-invariant) is almost generic (excess 120 at m = 9). Note that semicontinuity bounds
rank(σ') only from above, so this does not by itself bound e_m.

### 5.5 Conditional vanishing of the excess  [Proved]
**Theorem.** Fix m and C > 0, and suppose
  (H1_C)  dim ker(R_σ' | (u_m)_n) ≤ C·h^{(m)}_{n − k_m}  for all k_m ≤ n ≤ mid.
Then  e_m ≤ 2C·P(X_m ≤ mid − k_m) ≤ 2C·exp(−2(k_m + 3/2)² / Σ_{k<m} k² d_k).
If (H1_C) holds and k_m ≥ d(m) − 1 (Conjecture §3.4) for infinitely many m, then **c_e = 0**. Under these hypotheses,
e_m ≤ 2C·exp(−m/4 + O(1)) along those m.
*Proof.* By 5.1 and ker_n = 0 for n < k_m: e_m|G_m| ≤ 2C Σ_{n = k_m}^{⌊mid⌋} h_{n−k_m} = 2C|G_m|·P(X_m ≤ ⌊mid⌋ − k_m). Here
X_m = Σ_ℓ deg(ℓ)·B_ℓ over the N_m letters, with B_ℓ independent Bernoulli(1/2) (the Hilbert series is ∏(1+z^k)^{d_k}),
E X_m = top/2, and ⌊mid⌋ − k_m ≤ top/2 − (k_m + 3/2). Hoeffding's inequality for summands with ranges deg(ℓ) gives the
exponential bound. Next, d(m) ≥ Σ_{k<m, 3∤k} k = m²/3 − O(m) and Σ_{k<m} k² d_k = (8/9)m³ + O(m²), so the exponent is
2(m²/3)²/((8/9)m³) + O(1) = m/4 + O(1). Finally c_e ≤ x_m/3 = (B^Q_m/|G_m| + e_m)/3, and x_m is nonincreasing (§3.1(b)), so
c_e ≤ inf_m x_m/3. Along the given sequence, B^Q_m/|G_m| → 0 (§4) and e_m → 0. ∎
[Verified] (H1_3) holds for m = 5..10, with max_n ker_n/h_{n−k_m} = 3 exactly (attained at n = k_m) and decreasing afterwards
(m = 10: 0.90 at n = mid). k_m ≥ d(m) − 1 holds for m ≤ 11. Numerically 6·P(X_m ≤ mid − k_m) = 0.0044, 0.0044, 0.0139 at
m = 8, 9, 10, against e_m = 0.0022, 0.0027, 0.0041. The Hoeffding form is about 10 times weaker.
*Remark.* The central filtration (§3.1) cannot replace (H1). Its inequality moves the whole level-m kernel, including the large
maximal-rank part of the upper half, into the middle degrees of level m+1. Iterating it gives only x_{m+1} ≤ x_m. A strict
decrease would need a lower bound for the rank of the connecting map L_m ⊗ ker R^{(m)} → L_m ⊗ coker R^{(m)}, which was not obtained.

### 5.6 Conjectures
* [Conjecture] (H1_3) holds for all m, i.e. the lower-half kernel grows no faster than three copies of u_m shifted to k_m.
  Together with §3.4 this gives e_m → 0 and c_e = 0.
* [Conjecture/Heuristic] e_m ≈ c·P(X_m ≤ mid − k_m) with c ∈ [1.8, 3.8] (fitted for m = 5..10, `taskD4_bound.py`). This
  forecasts e_11 ≈ 0.002–0.0045 and e_12 ≈ 0.001–0.002, then decay like e^{−m/4} with bumps where d(m) has plateaus
  (m = 10, 16, 19, 22, 28, ...). The rise of e_m for m = 8 → 10 is the first such bump (k_10 = k_9 + 1 while mid grows by 9).
  Rigorous for m = 11 (`taskD5_analyze.py`): 0.00051 ≤ e_11 ≤ 0.01693.
