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

---------------------------------------------------------------------------------------------------
## 6. The threshold k_m (Task E)

Notation. U := u(L^s), the restricted enveloping algebra of n̂ in all degrees, so u^s_m = U/⟨L^s_{≥m}⟩ and (u^s_m)_j = U_j for
j < m. L_m := L^s/L^s_{≥m}. For a ≤ m, I_a := L^s_{≥a}/L^s_{≥m}, an ideal of L_m with L_m/I_a = L_a. ker^{(m)}_n := ker(R_σ' | (u^s_m)_n),
and ker^{(m)}_γ for a Q-block. A := Σ_j u ω_j (§5.2). P(m) is the statement "k_m = d(m) and ker^{(m)}_{d(m)} = span{ω_0, ω_1, ω_2}".
(H2) is the statement k_m ≥ d(m) − 1.

**Status.** (H2) is **not proved**, neither for all m nor for an infinite subsequence, and no bound k_m ≥ f(m) with f → ∞ is
proved. [Update, Task F, §7: R_σ' is injective on U, so k_m ≥ m − 3 → ∞ is now proved. (H2) itself remains open.] What is proved: the weakest form of (H2) that the conditional theorem can use (6.1); a dichotomy for the growth of
k_m (6.2); a window lemma that reduces (H2) to the "jump steps" m ∈ 𝓗_{m+1} (6.3); the connecting map at a jump step and its
closed form on the column integrals ω_j (6.4); an exact lifting method (6.5). Verified: k_12 = d(12) = 57 and P(12) (6.6).
Section 6.7 lists the approaches that failed and why.

### 6.1 What the conditional theorem needs from k_m  [Proved]
σ_m² := Var X_m = ¼ Σ_{k<m} k² d_k = (2/9) m³ + O(m²), because Σ_{k<m} k² d_k = 3 Σ_{k<m} k² − Σ_{3|k<m} k² = (1 − 1/9) m³ + O(m²).
For C > 0 and an integer κ ≥ 0, consider the hypothesis
   (H1'_{C,κ})  dim ker(R_σ' | (u_m)_n) ≤ C·h^{(m)}_{n−κ} for all n ≤ mid  (h_j := 0 for j < 0).
For κ ≤ k_m, this follows from (H1_C) of §5.5 only when h is nondecreasing on [0, mid]. For κ = k_m it is exactly (H1_C)
together with the definition of k_m.

**Proposition 6.1.** (a) Under (H1'_{C,κ}) at level m: e_m ≤ 2C·P(X_m ≤ ⌊mid⌋ − κ) ≤ 2C·exp(−(κ + 3/2)²/(2σ_m²)). Under
(H1_C) the same bound holds for every κ ≤ k_m.
(b) Hence **c_e = 0 follows from**: an infinite set 𝓜 of levels and numbers κ_m with (H1'_{C,κ_m}) for m ∈ 𝓜 (C fixed) and
**κ_m / m^{3/2} → ∞ along 𝓜**. Examples: κ_m = c·m² gives e_m ≤ 2C·exp(−(9/4)c²·m + O(1)), and c = 1/3 gives the e^{−m/4} of §5.5.
κ_m = m^{3/2}ψ(m) with ψ → ∞ gives e_m ≤ 2C·exp(−(9/4)ψ(m)²(1 + o(1))).
(c) This is sharp for the method. If k_m ≤ A·m^{3/2} along 𝓜, then P(X_m ≤ ⌊mid⌋ − k_m) ≥ Φ(−3A/√2) − o(1), so 2C·P_m does
not tend to 0. So (H2) can be weakened from k_m ≳ m²/3 to **k_m = ω(m^{3/2}) on a subsequence carrying (H1)**, but not to
k_m = O(m^{3/2}). In particular a linear bound cannot be used.
*Proof.* (a) As in §5.5: e_m|G_m| ≤ 2 Σ_{n≤mid} ker_n ≤ 2C Σ_{n ≤ ⌊mid⌋} h_{n−κ} = 2C|G_m|·P(X_m ≤ ⌊mid⌋ − κ). Under (H1_C) one
gets P(X_m ≤ ⌊mid⌋ − k_m), and t ↦ P(X_m ≤ t) is nondecreasing. Hoeffding's inequality for X_m = Σ deg(ℓ)B_ℓ, with
Σ (ranges)² = Σ k²d_k = 4σ_m² and deviation t = κ + 3/2 (since ⌊mid⌋ − κ ≤ E X_m − t), gives exp(−2t²/(4σ_m²)).
(b) For κ = c m²: t²/(2σ_m²) = c²m⁴/((4/9)m³)·(1 + O(1/m)). Then c_e ≤ inf x_m/3 and x_m = B^Q_m/|G_m| + e_m with
B^Q_m/|G_m| → 0 (§4), as in §5.5.
(c) The B_ℓ are independent and max deg(ℓ) = m − 1 = o(σ_m), so Lindeberg's CLT holds: (X_m − E X_m)/σ_m ⇒ N(0,1), uniformly
in the argument (Pólya). Also ⌊mid⌋ − k_m ≥ E X_m − k_m − 2 and (k_m + 2)/σ_m ≤ 3A/√2 + o(1). ∎

### 6.2 A dichotomy for the growth of k_m  [Proved; the constant 57 by computation]
**Proposition 6.2.** The following are equivalent: (i) k_m → ∞; (ii) R_σ' is injective on U (in every degree); (iii) k_m ≥ m − 3
for every m. If they fail, k_m is eventually constant, equal to the least degree K of a nonzero y ∈ U with yσ' = 0, and
K ≥ 57.
*Proof.* (ii)⇒(iii): for n ≤ m − 4 the map R^{(m)} : (u_m)_n → (u_m)_{n+3} equals R_σ' : U_n → U_{n+3}. (iii)⇒(i) is clear.
(i)⇒(ii): a nonzero y ∈ U_n with yσ' = 0 stays nonzero in u_m for m > n, so k_m ≤ n for all m > n. If (ii) fails with least
degree K, then k_m ≤ K for m > K. By (ii)⇒(iii) applied in degrees < K, k_m ≥ K for m ≥ K + 4. Finally K ≥ k_12 = 57 (§6.6),
because k is nondecreasing. ∎
So even an unconditional *linear* bound is equivalent to an injectivity statement in the infinite-dimensional U. That
statement is not proved, and by 6.1(c) a linear bound would not suffice anyway.

### 6.3 Window lemma; reduction of (H2) to the jump steps  [Proved]
**Lemma 6.3 (window).** Let a ≤ m and n < k_a + a. Then the reduction π_a : u_m → u_a (quotient by the ideal generated by I_a)
is injective on ker^{(m)}_n, blockwise, with image in ker^{(a)}_n.
*Proof.* With a PBW order putting the I_a-letters first, u_m = u(I_a) ⊗ V is a free left u(I_a)-module. Moreover
F^i := (u(I_a)^+)^i u_m = (u(I_a)^+)^i V is a finite decreasing filtration by right ideals, with gr^i = gr^i u(I_a) ⊗ u_a and
gr R_σ' = 1 ⊗ R^{(a)}. This is the argument of §3.1, which uses only that I_a is an ideal. gr^i u(I_a) lives in degrees ≥ i·a.
If 0 ≠ y ∈ ker^{(m)}_n lay in F^i \ F^{i+1} with i ≥ 1, its leading part would be a nonzero element of gr^i u(I_a) ⊗ ker R^{(a)}
whose u_a-components have degree ≤ n − a < k_a. That is impossible, so y ∉ F^1 = ker π_a. ∎
**Corollary 6.4.** (a) *(Free steps.)* If m ∉ 𝓗_{m+1} (3 | m, m ≠ 3·2^r), then d(m+1) = d(m). Hence k_m ≥ d(m) − c implies
k_{m+1} ≥ d(m+1) − c, and P(m) implies P(m+1).
(b) *(Reduction.)* (H2) holds for all m iff for every m ∈ 𝓗_{m+1} (3 ∤ m or m = 3·2^r) the threshold jumps by at least
m − 1: k_{m+1} ≥ d(m) + m − 1. Similarly, k_m = d(m) for all m iff the jump is at least m at every such step. At a jump step
d(m+1) = d(m) + m, while k_{m+1} ≤ d(m+1) always (Thm 2.4).
(c) For n < k_m + m we have ker^{(m+1)}_n ↪ ker^{(m)}_n, and ker^{(m+1)}_n = 0 as soon as the connecting map δ_m of 6.4 is
injective on ker^{(m)}_n. So k_{m+1} ≥ min(k_m + m, least n where δ_m is not injective on ker^{(m)}_n).
*Proof.* (a) The first claim is monotonicity. For P: by 6.3 with a = m, ker^{(m+1)}_n ↪ ker^{(m)}_n for n < d(m) + m, which is
0 for n < d(m). At n = d(m) the image lies in span{ω^{(m)}_j}, and it is all of it: since 𝓗_{m+1} = 𝓗_m, the elements
ω^{(m+1)}_j (Thm 2.4 at level m+1) reduce to ω^{(m)}_j. (b) Induction on m from k_2 = d(2) = 1, using (a) at the free steps.
(c) 6.3 and 6.4 below. ∎
So (H2) is a statement about the jump steps only: at each m ∈ 𝓗_{m+1}, the entire kernel of level m in the window
[k_m, k_m + m) must die when the central letters L_m are adjoined.

### 6.4 The connecting map at a jump step  [Proved]
For x ∈ ker^{(m)}_n choose a lift x̂ ∈ u_{m+1}. Then x̂σ' ∈ J := L_m u_{m+1} (L_m is central with zero 2-map, J/J² = L_m ⊗ u_m), and
        δ_m(x) := class of x̂σ' in L_m ⊗ coker(R^{(m)})_{n+3−m}.
This is well defined, since Jσ' ≡ L_m ⊗ im R^{(m)} mod J². It is **left u_m-linear**, δ_m(zx) = z·δ_m(x), because L_m is
central and coker R^{(m)} = u_m/u_mσ' is a left module. If x is the reduction of some y ∈ ker^{(m+1)}_n, then δ_m(x) = 0.
**Detection criterion.** v ∉ u_mσ' ⟺ v·w ≠ 0 for some w ∈ ker L_σ' := {w : σ'w = 0}. Indeed im R = (ker L_σ')^⊥ for the
Frobenius pairing λ(ab) (§1.6(iv)), and ker L_σ' is a right ideal.

**Proposition 6.5 (δ on the column integrals).** Let m ∈ 𝓗_{m+1}, column j = 0 (the others follow by ρ),
X := ∏_{k<m, 3∤k} c_k, T := c_3. Then c_{3·2^r} = T^{2^r}; write T^a := ∏_{r : bit r of a is 1} c_{3·2^r}, R := #{r : 3·2^r < m},
N := 2^R − 1, so ω_0 = X·T^N. With the lift ω̂_0 := X·T^N ∈ u_{m+1}:
        ω̂_0·σ' = c_m·ζ_0,   ζ_0 = X·T^{N−⌊m/3⌋}·(y1.1 y1.2 + y2.2)  (m ≡ 1 mod 3),
                               X·T^{N−⌊m/3⌋}·y1.2              (m ≡ 2 mod 3),
                               X                               (m = 3·2^R).
Hence δ_m(z ω_0) = c_m ⊗ [z ζ_0] for all z ∈ u_m.
*Proof.* In L_{m+1}: [c_k, c_3] = c_{k+3} for 3 ∤ k (§2.1). Real column letters commute and square to 0, and c_m is central.
So x T = T x + D(x) for x ∈ V := span{c_k : 3 ∤ k}, with D(c_k) = c_{k+3}, and T^N x = Σ_i C(N,i) D^i(x) T^{N−i}, where every
C(2^R − 1, i) is odd. By §1.5, σ' = c_1·A + c_2·y1.2 + c_3 with A = y1.1 y1.2 + y2.2. Now ω̂c_1 = Σ_{i≤N} X·c_{1+3i}·T^{N−i}. Here
X·c_{1+3i} = 0 if 1 + 3i < m (repeated real letter) and c_{1+3i} = 0 if 1 + 3i > m. The value 1 + 3i = m occurs iff m ≡ 1, with
i = ⌊m/3⌋ ≤ N because 3·2^R ≥ m; this term is X c_m T^{N−i} = c_m X T^{N−i}. In the same way ω̂c_2 = [m ≡ 2]·c_m X T^{N−⌊m/3⌋}.
Finally ω̂c_3 = X T^{N+1} = X c_{3·2^R}, which vanishes in u_{m+1} unless 3·2^R = m. ∎
[Machine check of the identity for m = 3..17, m ∈ 𝓗_{m+1}: `taskE_zeta2.py`, `out_taskE_zeta2.txt`.]

*Consequences and limits.*
(i) [Proved] span{ω_j} dies at the jump step m → m+1 iff ζ_j ∉ u_mσ' for each j. For 3 ∤ m the c^{(j)}_m are independent; for
m = 3·2^R they satisfy c^{(0)}_m + c^{(1)}_m + c^{(2)}_m = 0, but the ζ_j lie in different Q-blocks. Whether ζ_j ∉ u_mσ' holds
for all m is **open**. It holds for m ≤ 11, since k_{m+1} > d(m) there.
(ii) [Verified] ζ_0·Θ'(ω_{j'}) = 0 for all j' when m = 7, 8, 10, but not for m = 3..6 (`taskE_zeta.py`, `out_taskE_zeta.txt`).
Θ'(A) = Σ_j Θ'(ω_j)·u ⊂ ker L_σ' is exactly the annihilator-type part of ker L_σ' (§5.2 transported by Θ'). Therefore, for
m ≥ 7, every witness w ∈ ker L_σ' with ζ_0 w ≠ 0 is of cancellation type. Such witnesses exist for m ≤ 11. **A proof of the
jump, even restricted to the ω's, must use cancellation-type kernel elements (η-like) of the dual problem.** The
annihilator structure (§2, §5.2) alone cannot give it. This is the main obstruction met in Task E.
(iii) [Verified] The m = 9 element sits at the edge of the window. η ≡ y7.0·ω^{(8)}_2 = c^{(0)}_7·ω^{(8)}_2 (mod L_8), of degree
d(8) + 7 = d(9) − 1, and η = c^{(0)}_7·ω̂_2 + y8.0·w with wσ' = c^{(0)}_7·ζ_2 at level 8 (y8.0 = c^{(2)}_8). So δ_8 restricted to
A^{(8)} is *not* injective on the window; it fails on c^{(0)}_7ω_2. η then dies at the free step 9 → 10 (δ_9(η) ≠ 0;
ker^{(10)}_35 = 0, recomputed in 6.5).

### 6.5 Exact computation of ker R^{(M)} from level a by layer-wise lifting  [Proved method; validated]
Let 2a ≥ M. Then I := I_a is abelian with zero 2-map, u(I) = Λ(I), F^i := span of the PBW monomials with ≥ i I-letters
(I-letters first), gr^i = Λ^i(I) ⊗ u_a, and gr R = 1 ⊗ R^{(a)}.
**Lemma 6.6.** For a map T compatible with a finite filtration, the leading part of every y ∈ ker T ∩ F^i lies in
ker(d_1 | E_1^i), where E_1^i := ker(gr^i T) and d_1(x) := [T(x̃)] ∈ coker(gr^{i+1} T). Hence dim ker T ≤ Σ_i dim ker(d_1|E_1^i).
(Use y itself as the lift x̃.) Here E_1^i = ⊕_{|S|=i} S ⊗ ker R^{(a)}_{γ−wt S} and d_1(S ⊗ x) = Σ_ℓ (S∪ℓ) ⊗ [z_ℓ], where
(S x̂)σ' ≡ Σ_ℓ (S∪ℓ)·z_ℓ mod F^{i+2}.
*Exact algorithm* (`taskE_lift.py`). Solve yσ' = 0 layer by layer. The initial candidates are a basis of ker R^{(a)}_γ. At
layer i, the layer-i components of the candidates' products are reduced modulo ⊕_{S'} S' ⊗ im R^{(a)}, using an echelon form
of [R^{(a)} | Id] that also returns preimages. A basis of the combinations with zero normal form is corrected by the preimages
S'·p̂, and then S ⊗ ker R^{(a)} (|S| = i) is added. At the end every candidate is checked to satisfy yσ' = 0. Completeness: a
kernel element minus a suitable combination of candidates has its lowest nonzero layer i in ⊕_{|S|=i} S ⊗ ker R^{(a)}.
So the final candidates form a basis of ker R^{(M)}_γ.
[Validated] It reproduces the directly computed kernels exactly: M = 9 from a = 8 (n = 28..38, including η at n = 35);
M = 10 from 9 (n ≤ 40, including the death of η); M = 11 from 9 and from 10 (n ≤ 47: 0 for n ≤ 45, then 1 and 2 per ρ-orbit
at n = 46, 47). The plain E_2 bound of Lemma 6.6 is not sharp: M = 11 from 10 gives E_2 = 1 and 3 at n = 45, 46, where the
true values are 0 and 1. Higher layers matter.

### 6.6 m = 12  [Verified]
`taskE_lift.py 12 10 46 51` and `… 52 57` (1600 s in total) give ker^{(12)}_n = 0 for 46 ≤ n ≤ 56; degrees ≤ 45 are covered by
§3.1(c). ker^{(12)}_57 is one vector per ρ-orbit, in block (15,23,19) and its rotations, namely the single PBW monomial
y11.2·y10.1·y8.2·y7.1·y6.0·y5.2·y4.1·y3.0·y2.2·y1.1 = ω^{(12)}_1. Hence **k_12 = d(12) = 57 and P(12) holds**: no η-type
defect at m = 12 = 3·2². Up to m = 12: k_m = d(m) for m = 2..8 and 10, 11, 12, and k_9 = d(9) − 1. With 6.2, either
k_m ≥ m − 3 for all m, or k_m is eventually constant with value ≥ 57.

### 6.7 Approaches that do not work, and why
(a) *Weight filtrations of u_m* [Proved]. Take any weights on the letters for which the PBW-weight filtration is
multiplicative (w([x,y]) ≥ w(x) + w(y), descending; or ≤, ascending). Then the leading form of σ' is never the single Cartan
letter y3.1 (or c_3): [y1.0, y2.2] = y3.1 forces the term y1.0·y2.2 to be at least as leading. So one cannot reduce to R_{c_3}.
That map is injective on U (U is free over F2[c_3] = u(span{c_{3·2^r}})), but on u_m its kernel already starts in degree
3(2^R − 1) = O(m).
(b) *Induced modules* [Proved inequality, numerically useless]. For any restricted subalgebra K: dim ker^{(m)}_n ≤
Σ_μ dim gr u(K)_μ · dim ker(σ' | F2 ⊗_{u(K)} u_m)_{n−μ}, so k_m ≥ k(F2 ⊗_{u(K)} u_m). The proof is that of 6.3, with the right
ideals (u(K)^+)^i u_m. K = I_a gives k_m ≥ k_a. For non-ideal K the bound collapses: m = 6 (k = 15) gives 12 for a Cartan tower,
8 for row reals, 6 for row subalgebras and 3 for upper-triangular real letters; m = 7 (k = 21) gives 12 for the tower and 9 for a
row subalgebra (`taskE_induced.py`).
(c) *Monomial orders* [Verified]. For lexicographic PBW orders (highest- or lowest-degree letters dominant, max or min), the
leading-monomial map w ↦ lead(wσ') is not injective already in degree 2 (m = 6, `taskE_lead.py`). So no triangularity for
such orders.
(d) *Annihilator witnesses* fail from m = 7 on (6.4(ii)). *Powers of σ'*: σ' has nilpotency index 3, 5, 10, 12, 16, 24, 31
for m = 3..9, and σ'^{e−1} lies within 9 of the top degree, so ζσ'^{e−1} = 0 for degree reasons and gives no test.
(e) *Two-level E_2 bounds* are not sharp (6.5), and the central filtration alone transports kernels without killing them (§5.5).
(f) *Frobenius doubling* Fr : e_p ↦ e_{−p}, φ ↦ φ² is a ring endomorphism of A^s, giving u_m → u_{2m} with degrees doubled.
But Fr(σ') = y6.1 + y4.2·y2.1 + y2.0·y2.1·y2.2 ≠ σ'² and is unrelated to σ'. Even if it were related, it would only give
k_{2m} ≥ 2k_m, a linear bound.

### 6.8 Refined conjecture
[Conjecture] k_m = d(m) − 1 if 3 | m, m ≥ 9 and m ≠ 3·2^r; otherwise k_m = d(m) and P(m) holds. The defect at such m is a
lift of the window-edge element c^{(j')}_{m−2}·ω^{(m−1)}_j, as for η at m = 9 (6.4(iii)), and it dies at the next (free)
level. Evidence: m ≤ 12 (one defect level, m = 9; m = 3, 6, 12 = 3·2^r have none, as predicted). The next informative
levels are m = 13, 14 (predicted 69, 82) and m = 15 (predicted 95 = d(15) − 1). With the present methods they are out of
reach: going from a = 10 or 11 to M ≥ 13 brings the upper-half kernel of level a into E_1.

---------------------------------------------------------------------------------------------------
## 7. Cancellation-type kernel elements (Task F)

Notation as in §6: U = u(L^s) = u(n̂) over F2, u_m = U/I_m with I_m := U·L^s_{≥m} (two-sided ideal), π_a : u_m → u_a.
e_i := y1.i (i = 0, 1, 2) are the Chevalley generators. Scripts: `taskF_*.py`; numbers in RESULTS, "Task F".

**Summary.** (1) **R_σ' is injective on U** (Theorem 7.1). By 6.2 this gives the first unconditional bound tending to
infinity, **k_m ≥ m − 3 for all m**. It also shows that σF is a non-zero-divisor in u(L) = gr F2[[P]], and that F is a
non-zero-divisor in F2[[P]]. (2) The proof uses tensor products of dual evaluation modules. On them σ' acts with nonzero
determinant, but the inverse has non-monomial denominators. That is exactly why the method gives no superlinear bound
by itself (7.2). (3) A quadratic bound k_m ≳ m²/6 reduces to one divisibility statement per level m (7.3):
ω_{K} ∈ σ'·u_m, where ω_K is the product of the 8 letters of degrees m−3, m−2, m−1. This statement is equivalent to
"every kernel element of level m dies in u_{m−3}". It is verified for m ≤ 10 in all degrees, but not proved. (4) Structural
data on kernel elements, including η, and the approaches that failed (7.4, 7.5).

### 7.1 Injectivity on U  [Proved; machine checks of the ingredients in `taskF_faithful.py`]
**Lemma 7.1a (cyclic form).** In U, σ' = e0e1e2 + e1e2e0 + e2e0e1.
*Proof.* By §1.5, σ' = y1.0y1.1y1.2 + y1.2y2.1 + y1.0y2.2 + y3.1, with y2.1 = [e1, e0], y2.2 = [e2, e1] and
y3.1 = [e0, [e2, e1]] (bracket formula of §1.2: e.g. [y1.0, y2.2] = e_0φ³ + e_2φ³ = y3.0 + (y3.0 + y3.1)). Expanding over F2:
y1.2y2.1 = e2e1e0 + e2e0e1, y1.0y2.2 = e0e2e1 + e0e1e2, y3.1 = e0e2e1 + e0e1e2 + e2e1e0 + e1e2e0. Summing with e0e1e2 gives
e0e1e2 + e1e2e0 + e2e0e1. ∎ [also machine-checked, `taskF_words.py`]

**Lemma 7.1b (the dual natural representation).** Let π : A^s → M_3(F2[t]), D_kφ^k ↦ D_kΠ^k (§1.3), and
ρ*(x) := (π(x) + tr(π(x))·I)^T. Then ρ* is an injective homomorphism of graded restricted Lie algebras
L^s → gl_3(F2[t]). Here deg(E_pq t^k) := 3k + q − p, and the 2-map is the matrix square. It extends to an algebra map
ρ* : U → M_3(F2[t]) with ρ*(e0) = tE20, ρ*(e1) = E01, ρ*(e2) = E12 and **ρ*(σ') = t·I**.
*Proof.* x ↦ π(x) + tr(π x)I is a Lie map (traces of commutators vanish), kills T^s (π(t^j) = t^jI and 3 = 1), is injective
on L^s (π(x) scalar ⇒ x ∈ T^s, §1.3), and respects squares because tr(M²) = tr(M)² in characteristic 2.
Transposition is an anti-automorphism, hence a restricted Lie automorphism in characteristic 2 (§1.4). Grading: the
transpose of E_ij t^n (principal degree 3n + i − j) is E_ji t^n. Finally E20E01E12 = E22, E01E12E20 = E00 and
E12E20E01 = E11, so by 7.1a ρ*(σ') = t(E22 + E00 + E11) = tI. ∎ [machine check: graded, injective on letters of degree < 13,
homomorphism on 300 random products, ρ*(σ') = tI; `out_taskF_faithful.txt`]

For r ≥ 1 put Ψ_r := (ρ*_{t_1} ⊗ ⋯ ⊗ ρ*_{t_r}) ∘ Δ^{(r)} : U → M_3(F2[t_1]) ⊗ ⋯ ⊗ M_3(F2[t_r]) = M_{3^r}(F2[t_1,…,t_r]),
the action on V*(t_1) ⊗ ⋯ ⊗ V*(t_r). It is an algebra homomorphism.

**Lemma 7.1c (determinant).** det Ψ_r(σ') is a nonzero polynomial. More precisely det Ψ_r(σ')(t, 0, …, 0) = t^{3^r}.
*Proof.* Induction on r; r = 1 is 7.1b. By coassociativity Ψ_r = (Ψ_{r−1} ⊗ ρ*_{t_r}) ∘ Δ. Write Δ(σ') = σ'⊗1 + Σ a_i ⊗ b_i
with b_i ∈ U^+, which holds for any element of a connected Hopf algebra. At t_r = 0, ρ*_0 maps every letter to a strictly
upper triangular matrix: only letters with t-free matrix survive, namely y1.1, y1.2, y2.2 ↦ E01, E12, E02, and Cartan
letters go to 0. So ρ*_0(U^+) consists of strictly upper triangular matrices N_i. Hence
Ψ_r(σ')|_{t_r=0} = Ψ_{r−1}(σ') ⊗ I + Σ Ψ_{r−1}(a_i) ⊗ N_i. In 3×3 block form indexed by the last tensor factor, this is
block upper triangular with diagonal blocks Ψ_{r−1}(σ'). So det Ψ_r(σ')|_{t_r=0} = (det Ψ_{r−1}(σ'))³. ∎
[`taskF_faithful.py` (3): t^{3^r} at (t,0,…,0) for r ≤ 5; `taskF_tensor.py`: full rank at random points of GF(2^16)^r, r ≤ 6.]

**Lemma 7.1d (faithfulness).** For every 0 ≠ y ∈ U_n there is r ≤ n with Ψ_r(y) ≠ 0.
*Proof.* Let C = ⊕_d (U_d)^* be the graded dual. It is a connected graded commutative algebra (U is cocommutative), with
(fg)(y) = (f⊗g)(Δy). Let X ⊂ C be the span of the matrix-coefficient functionals φ_{pq,k}(y) := coefficient of t^k in
ρ*(y)_{pq}. It is a graded subspace by 7.1b, and contains ε = φ_{00,0}. A product φ_1⋯φ_r evaluated at y is a matrix entry
and t-coefficient of Ψ_r(y). So it suffices that X generates C as an algebra. By graded Nakayama it suffices that
X ∩ C^+ → C^+/(C^+)² is onto. In degree d > 0, ((C^+)²)^⊥ ∩ U_d = P(U)_d, the primitives (Δ_{a,b}(y) = 0 for 0 < a < d), and
P(U) = L^s for a restricted enveloping algebra. So (C^+/(C^+)²)_d = (L^s_d)^*, and surjectivity means that no
0 ≠ x ∈ L^s_d is killed by all φ_{pq,k}, i.e. that ρ* is injective on L^s (7.1b). A nonzero functional of degree n is a
sum of products of at most n factors from X ∩ C^+; pad with ε. ∎ [numerical confirmation for n ≤ 6: `out_taskF_faithful.txt` (2)]

**Theorem 7.1.** R_σ' : U → U, y ↦ yσ', and L_σ' are injective. Equivalently (field extension, §1.6), R_{σF} and L_{σF}
are injective on u(L), in every degree.
*Proof.* Let 0 ≠ y ∈ U and choose r with Ψ_r(y) ≠ 0 (7.1d). Then Ψ_r(yσ') = Ψ_r(y)Ψ_r(σ'). Here Ψ_r(σ') is invertible over
F2(t_1,…,t_r) (7.1c) and Ψ_r(y) is a nonzero matrix, so the product is nonzero. The same argument works for L_σ'. ∎

**Corollary 7.2.** (a) **k_m ≥ m − 3 for every m**, so k_m → ∞ (Prop. 6.2, (ii) ⇒ (iii)). With §6.6 and monotonicity,
k_m ≥ max(57, m − 3) for m ≥ 12. (b) σF is a non-zero-divisor on both sides of gr F2[[P]] = u(L) (Task 2 identification).
(c) F is a non-zero-divisor on both sides of F2[[P]]. Indeed, if xF = 0 with x ≠ 0, the augmentation filtration is
separated, so x has a leading form gr x ≠ 0. Then gr(x)·σF ≠ 0 would be the leading form of xF = 0, a contradiction.
(This answers the question of Task 3 positively for σF. The "gr is a domain" route fails, but only regularity of σF
is needed.) [Proved; (b), (c) use the Jennings–Quillen identification gr F2[[P]] ≅ u(L) of Task 2.]

### 7.2 Why Theorem 7.1 alone gives only a linear bound  [Proved / Verified]
* A t-adic version of 7.1 works on the u_m-modules ⊗_s V*[t_s]/(t_s^K) (K ≈ m/3), where L_{≥m} acts trivially. It needs a
  polynomial B with Ψ_r(σ')B = D·I and D a **monomial** in the t_s. The determinant is not of this kind:
  Ψ_2(σ') = (t1 + t2)·I exactly, and det Ψ_3(σ') = s1^21 (s1³ + t1t2t3)², s1 = t1 + t2 + t3 (`taskF_symb.py`,
  `out_taskF_det3.txt`). In F2[t1,t2]/(t1^K, t2^K), multiplication by t1 + t2 has a large kernel. Concretely, for two
  strings of length ≥ 9 the element t1t2 is not in (t1+t2)·F2[x,y]/(x³,y³). So per-site truncation is lost.
* With a single variable (t_s = λ_s t, λ generic) the denominator is a monomial: σ' acts as t·Σ(λ) with Σ(λ) invertible.
  But these modules only see functionals of total t-degree < K, i.e. elements of degree ≲ m. This reproduces k_m ≳ m − O(1),
  no more.
* (Gröbner / nil-Coxeter view.) U is a quotient of F2⟨e0,e1,e2⟩/(e_i², (e_ie_j)² + (e_je_i)²): leading words 00, 11, 22,
  1010, 2020, 2121 in degrees 2 and 4, then words in degrees 6, 7, 8, 10, … (`out_taskF_words.txt`). σ' is the cyclic sum of
  the Coxeter word. The prefix-triangularity criterion "lead(wσ') = w·c" fails from degree 3 or 4 on, for each of the 6
  letter orders.
  So there is no Gröbner proof of 7.1 of this type.
* (Cartan witnesses.) If 0 ≠ z ∈ U σ' ∩ u(ĥ) (ĥ = span of all Cartan letters, u(ĥ) a polynomial ring and U free over it),
  injectivity would follow at once. But Uσ' ∩ u(ĥ) = 0 in weights kδ, k ≤ 6 (`taskF_cartan.py`). The proof in 7.1 instead
  uses non-graded modules.

### 7.3 A quadratic bound reduced to one divisibility per level  [Proved criterion; Verified m ≤ 10; Conjecture]
For 1 ≤ a < m let K_a := L^s_{[m−a, m)} (letters of degree m−a, …, m−1), a restricted ideal of L_m. Let ω_{K_a} be its
integral, the product of all its letters, and J_a := ker π_{m−a} = u_m·K_a.

**Proposition 7.3 (criterion).** ker R^{(m)} ⊂ J_a ⟺ ω_{K_a} ∈ σ'·u_m ⟺ ω_{K_a} ∈ u_m·σ'.
*Proof.* Use the Frobenius pairing λ(ab) on u_m (§1.6). ker R_σ' = ^⊥(σ'u_m). Next, J_a^⊥ = {b : K_a b = 0}, and this
equals ω_{K_a}·u_m by the PBW argument of §2.3 with the K_a-letters first: b = Σ f_i v_i, and K_a b = 0 iff each f_i is an
integral of u(K_a). Taking ⊥, ker R ⊂ J_a ⟺ ω_{K_a}u_m ⊂ σ'u_m ⟺ ω_{K_a} ∈ σ'u_m. Θ' fixes σ' and ω_{K_a}, which gives
the left version. ∎
Equivalently, the integral Λ_{m−a} of u_{m−a} (degree top(u_{m−a})) is not π_{m−a} of a kernel element of level m. Since a
nonzero left ideal of u_{m−a} contains Λ_{m−a}, π_{m−a}(ker R^{(m)}) ≠ 0 iff it contains Λ_{m−a}.

**Corollary 7.4.** If ω_{K_a} ∈ σ'u_m, then **k_m ≥ k_{m−a} + m − a**.
*Proof.* Window Lemma 6.3 with level m − a: for n < k_{m−a} + m − a, π_{m−a} is injective on ker^{(m)}_n, and it vanishes
there by 7.3. ∎

**Conjecture 7.5 (C3).** ω_{K_3} ∈ σ'·u_m for every m ≥ 6. That is, every kernel element of level m involves, in every PBW
term, a letter of degree ≥ m − 3.
*Evidence* [Verified]. (i) ω_{K_3} ∈ u_mσ' for m = 6, 7, 8, 9, 10 (`taskF_omegaK.py`; for m = 10 a rank computation in
a block of size 80871 → 77517). The same holds for a = 4 (m = 8, 9, 10). It fails for a = 2 when m = 7, 8, 9, 10, and for
a = 1 always. (ii) Directly: dim π_{m−3}(ker R^{(m)}_n) = 0 in **all** degrees n for m = 5..8, and for n ≤ 60 at m = 9
(`taskF_top.py`). π_{m−2}(ker) ≠ 0 only in the 7 degrees top(u_{m−2}) − 6 … top(u_{m−2}) (m = 7, 8, 9). (iii) All kernel vectors known
at m = 11 (n ≤ 47) and m = 12 (n = 57) even lie in J_1, and those at m = 9, 10 (n ≤ 38, 40) in J_2 (η included).
*Consequence (conditional).* Under C3 for all m ≥ m_0: k_m ≥ k_{m−3} + m − 3, hence **k_m ≥ m²/6 − O(m)**. Then
κ_m = k_m satisfies κ_m/m^{3/2} → ∞, which is the hypothesis of Prop. 6.1(b). Under (H1_C), e_m ≤ 2C·exp(−m/16 + O(1))
((m²/6)²/(2·(2/9)m³) = m/16) and c_e = 0. Even C3 along an arithmetic progression of steps, or the weaker C_a with a = a(m)
= O(m^{1/2−ε}), would suffice: iterating 7.4 gives k_m ≳ m^{2}/(2a).
*Why C3 is plausible and where a proof should come from.* Dually (C := U^*, D := σ'⇀ with (Df)(y) = f(yσ')), C3 says that
the top functional λ_{m−3} of u_{m−3} lies in D(C_m), C_m = u_m^* = I_m^⊥. On matrix coefficients D acts by the shift
φ_{pq,k} ↦ φ_{pq,k−1} (because ρ*(σ') = tI). So it raises the level by one unit of t, i.e. by 3 principal degrees, which
is where the "3" comes from. The obstruction to turning this into a proof is the non-monomial denominator of 7.2. The
3-step lift from level m−3 to m cannot be detected by the first connecting map: d_1(Λ_{m−3}) = 0 because R^{(m−3)} is onto
in the degrees within k_{m−3} of the top. So higher layers of the spectral sequence of §6.5 are involved.

### 7.4 Cancellation-type elements: what is known  [Verified]
* η (m = 9) lies in J_2 (every PBW term has a letter of degree 7 or 8) but not in J_1, in accordance with C3. The ω_j lie
  in J_2 always: they contain c_{m−1} or c_{m−2}. More generally every kernel vector examined lies in J_2: all of the
  lower half for m ≤ 9, and the lifted kernels of §6.5 for m = 10, 11, 12.
* The preimages w with wσ' = ω_{K_3} have no closed form in the data. m = 6: 5 PBW terms, each ω_K with one or two letters
  traded for lower column letters (`out_taskF_omegaK_show.txt`). m = 7: 47 terms, with 5–7 K-letters per term. They
  resemble the column carries of Prop. 6.5, applied to three columns at once.
* Cartan witnesses (7.2) and annihilator witnesses (6.4(ii)) do not exist. The working witnesses are the non-graded
  modules V*(t_1) ⊗ ⋯ ⊗ V*(t_r), on which σ' acts as (Σ t_s)·I + N. Here N = 0 for r = 2. For r = 3, N is supported on
  the six permutation vectors f_{π0}⊗f_{π1}⊗f_{π2}, where it consists of two 3-cycles of weight t1t2t3; it is not
  nilpotent.

### 7.5 Status of (H2) after Task F
* Proved: k_m ≥ m − 3 for all m (k_m → ∞). With the verified k_m (m ≤ 12) and monotonicity: k_m ≥ max(k_12, m − 3).
* Not proved: any superlinear bound. In particular (H2) and its weakest useful form k_m/m^{3/2} → ∞ (6.1) remain open.
  Both follow from Conjecture C3 (or C_a with a = O(m^{1/2−ε})) via 7.3–7.4, and C3 is a single divisibility statement per
  level.
  **[Update, Task H, §8: C3 is false for infinitely many m (it fails at least once in every 61 consecutive levels), and
  under (H2) it fails for every m ≥ 27. The hypothesis "C_a with a = O(m^{1/2−ε}) for all large m" is self-contradictory.
  Every bound obtainable by iterating 7.4 is ≤ b + 4m^{3/2}, so this route cannot reach k_m/m^{3/2} → ∞.]**

---------------------------------------------------------------------------------------------------
## 8. Quadratic lower bound (Task H)

Notation as in §6–7. For 1 ≤ a < m: K_a := L^s_{[m−a, m)} ⊂ L_m (letters of degree m−a, …, m−1; a restricted ideal),
J_a := u_m K_a = ker π_{m−a}, **T_a(m) := Σ_{k=m−a}^{m−1} k·d_k = top degree of u(K_a)**, D := top(u_{m−a}), Λ_{m−a} the
integral (top monomial) of u_{m−a}. C_a(m) is the statement ω_{K_a} ∈ σ'u_m (⟺ ker R^{(m)} ⊂ J_a, Prop. 7.3).
k^R, k^L: thresholds of R_σ', L_σ'; on u_m they coincide (Θ', §1.4). Scripts `taskH_*.py`; numbers in RESULTS "Task H".

**Summary.** The goal (k_m ≥ c·m², or k_m/m^{3/2} → ∞ on an infinite set) was **not** reached. The main finding is
negative and proved: **the Task F route cannot work.** The statement C_a(m) forces k_{m−a} ≤ T_a(m) − 3 ≈ (8/3)·a·m
(Theorem 8.1). So C_a fails as soon as the threshold is superlinear, C3 is false for infinitely many m, and every lower
bound obtainable by iterating Corollary 7.4 (with any steps a(m)) is O(m^{3/2}) (Theorem 8.3). That is exactly the
borderline that Prop. 6.1(c) shows to be useless. Any successful argument has to be degree-sensitive (8.5).

### 8.1 Forced lifting near the top  [Proved; verified on 9 pairs (m, a), `taskH_lift_check.py`]
**Theorem 8.1.** Let γ be a Q-block, and let Ŝ be the set of Q-weights of the nonempty PBW monomials in the K_a-letters.
If coker R^{(m−a)} = 0 in every block γ + δ − μ, μ ∈ Ŝ, then π_{m−a}(ker R^{(m)}_γ) = ker R^{(m−a)}_γ: every kernel
vector of level m − a lifts to a kernel vector of level m.
In principal degrees: π_{m−a}(ker^{(m)}_n) = ker^{(m−a)}_n for every n > D + T_a(m) − 3 − k_{m−a}.
In particular, **if T_a(m) ≤ k_{m−a} + 2, there is y ∈ ker R^{(m)} of degree D with π_{m−a}(y) = Λ_{m−a}, so C_a(m)
fails.** Equivalently: **C_a(m) ⟹ k_{m−a} ≤ T_a(m) − 3.**
*Proof.* F^i := (u(K_a)^+)^i u_m is a finite decreasing filtration by right ideals, with F^1 = J_a. With the K_a-letters
first in the PBW order, F^i = (u(K_a)^+)^i ⊗ V with V ≅ u_{m−a} via π, gr^i = gr^i u(K_a) ⊗ u_{m−a}, and
gr R_σ' = 1 ⊗ R^{(m−a)} (§6.3; only the ideal property of K_a is used). Everything is Q-graded, and the weights of
gr^{≥1} u(K_a) are the weights in Ŝ (same Hilbert series as the PBW monomials). By hypothesis 1 ⊗ R^{(m−a)} maps onto
gr^i_{γ+δ} for every i ≥ 1. A map compatible with a finite filtration whose associated graded map is onto is onto, so
R_σ' : (J_a)_γ → (J_a)_{γ+δ} is onto. Let x ∈ ker R^{(m−a)}_γ and let x̂ ∈ (u_m)_γ be a lift. Then π(x̂σ') = xσ' = 0, so
x̂σ' = zσ' with z ∈ (J_a)_γ, and y := x̂ − z ∈ ker R^{(m)}_γ satisfies π(y) = x.
Degree version: the blocks γ + δ − μ have degree n + 3 − e, where e = deg μ ∈ [m − a, T_a(m)]. By the Frobenius pairing
of u_{m−a}, coker R^{(m−a)} in degree n + 3 − e is dual to (ker L_σ')_{D−n−3+e} (detection criterion, §6.4), and this
vanishes when D − n − 3 + e < k^L_{m−a} = k_{m−a}. For n = D, x = Λ_{m−a} (Λσ' = 0 for degree reasons) and e ≤ T_a(m)
this is the condition T_a(m) − 3 < k_{m−a}. Then y ∉ J_a, so ker R^{(m)} ⊄ J_a, which is the negation of C_a(m) (7.3). ∎
[Verified] `taskH_lift_check.py` computes dim π_{m−a}(ker R^{(m)}_γ) exactly for every block (one per ρ-orbit) with
ker R^{(m−a)}_γ ≠ 0, for (m, a) = (6,1), (7,1), (7,2), (8,1), (8,2), (8,3), (9,1), (9,2), (9,3), (10,1), (10,2). On all
blocks where the hypothesis holds (1, 19, 0, 10, 0, 0, 29, 0, 0, 132, 0 blocks) lifting is complete. There are no violations.
*Remark.* ω_{K_a} is central in u_m. For x of positive degree, ad x is a derivation preserving u(K_a) (K_a is an ideal) and
raising degree, so it kills the top degree of u(K_a). This is the "tensor decomposition" of idea 2 in the task. It does not
help: σ' acts diagonally only on gr (σ' does not commute with K_a). That diagonal action, together with the surjectivity
of R^{(m−a)} near its top, is exactly what forces Theorem 8.1.

### 8.2 Consequences for C_a  [Proved; constants by computation `taskH_nogo.py`, `taskH_gaps.py`]
T_a(m) < 3am. More precisely T_3(m) ≤ 8m − 15, since one of m−3, m−2, m−1 is a multiple of 3, with d = 2.
**Corollary 8.2.**
(a) *(fixed a)* For each fixed a, C_a fails for infinitely many m. Indeed, C_a for all m ≥ M_0 gives k_m ≥ m²/(2a) − O(m) by
7.4. Then T_a(m) < 3am ≤ k_{m−a} + 2 for large m, contradicting 8.1.
(b) *(C3, quantitative, unconditional)* **Among any 61 consecutive levels M, …, M+60 there is one where C3 fails.** Proof:
assume C3 on [M, M+60] and put x = M + 57. Iterating 7.4 twenty times, at x, x−3, …, x−57 ≥ M, gives
k_x ≥ Σ_{i=0}^{19}(x − 3 − 3i) = 20x − 630. C3 at x + 3 requires k_x ≤ T_3(x+3) − 3 ≤ 8x + 6. These are incompatible
because x ≥ 60. The exact worst-case gap from `taskH_gaps.py` (M ≤ 3000, using the known k_m, m ≤ 14) is 43 for a = 3. The
analogous gaps are 4, 12, 43, 79, 132, 194 for a = 1..6 (24 for a = 3 when M ≥ 1500). So **Conjecture 7.5 (C3 for all
m ≥ 6) is false.** No specific failing level is known for a = 3. With proved values only, C_1 is excluded for
m = 7, …, 29, 31, 34, 37, 40 and C_2 for m = 16, 17 (k_14 = 82, k_15 ≥ 82).
(c) *(growing a)* If a(m) = o(√m), then C_{a(m)} fails for infinitely many m. In particular the hypothesis "C_a with
a = O(m^{1/2−ε})" of 7.5 is self-contradictory. Proof: assume C_{a(m)} for all m ≥ M_0 and put A(m) := max_{m/2≤j≤m} a(j).
The chain m → m − a(m) → … stays in [m/2, m] for ≥ m/(2A(m)) steps, each gaining ≥ m/2, so k_m ≥ m²/(4A(m)) − O(m). Applied
at m − a(m), this contradicts k_{m−a(m)} ≤ T_{a(m)}(m) − 3 < 3a(m)m once A(m)² < m/13. ∎
(d) *(under the conjectured size of k)* If k_j ≥ d(j) − 1 for all j (the conjecture (H2)), then C3 fails for **every**
m ≥ 27. Under the refined conjecture 6.8 it fails for every m ≥ 26 (checked up to 10^4; beyond that d(m−3) ≥ (m−4)²/3 − O(m)
≫ 8m). Under 6.8, C_a(m) is possible only for a ≥ a*(m), where a*(m)/m = 0.20, 0.15, 0.13, 0.11, 0.107 at m = 10, 20, 40,
100, 1000, and a*/m → 1 − √0.8 = 0.106. This is where (4/3)(2θ − θ²) = (1 − θ)²/3. With such a, 7.4 gives only
k_m ≥ k_{0.9m} + 0.9m: a linear bound.
The data of Task F are consistent with this: C3 holds for m ≤ 10 only because T_3(m) ≈ 8m still exceeds k_{m−3} there
(first excluded level 26 under 6.8).

### 8.3 Ceiling of the iterated route  [Proved; numerics `taskH_nogo.py`]
**Theorem 8.3.** Let m = m_0 > m_1 > … > m_s with C_{a_i}(m_i), a_i = m_i − m_{i+1}, and let b ≤ k_{m_s} be a proved base
value. Then the bound that Corollary 7.4 yields, B_0 := b + Σ_{i=1}^{s} m_i, satisfies **B_0 ≤ b + 4m^{3/2}**.
*Proof.* Put B_i := b + Σ_{j>i} m_j. Then k_{m_i} ≥ B_i by 7.4, and Theorem 8.1 at m_i gives
B_{i+1} ≤ k_{m_{i+1}} ≤ T_{a_i}(m_i) − 3 < 3a_i m_i. We show B_i ≤ b + 4m_i^{3/2} by downward induction (B_s = b).
If a_i ≤ √m_i, then B_i = m_{i+1} + B_{i+1} ≤ m_i + 3m_i^{3/2} ≤ 4m_i^{3/2}. If a_i > √m_i, then m_{i+1} ≤ m_i − √m_i and
B_i ≤ m_i + b + 4(m_i − √m_i)^{3/2} ≤ b + m_i + 4m_i^{3/2}(1 − m_i^{−1/2}) ≤ b + 4m_i^{3/2}, using (1−x)^{3/2} ≤ 1 − x. ∎
Since every proved base value is O(m) (Theorem 7.1, monotonicity, finitely many computed k_m), **the route of §7.3 cannot
prove k_m/m^{3/2} → ∞ along any sequence.** By Prop. 6.1(c), that is exactly what the application needs. The optimum over
all chains, G(m) = max(base, max_a [(m−a) + min(G(m−a), T_a(m) − 3)]), is G(m)/m^{3/2} = 1.24, 1.28, 1.31, 1.32 at
m = 100, 200, 1000, 4000 (G(m)/d(m) = 0.36, 0.26, 0.12, 0.06).
**Remark 8.4 (any chain of ideals).** The same budget argument applies to chains of letter sets ∅ = S_0 ⊂ … ⊂ S_N = all
letters of L_m, with each T_i = S_i \ S_{i−1} an ideal of span S_i, using the window lemma 6.3 and criterion 7.3 for T_i.
This needs the proved bounds G_i to hold for both thresholds k^R and k^L of u(span S_i), e.g. for θ-stable S_i. A step gains
mindeg T_i < m and needs top(u(T_i)) ≥ G_{i−1} + 3 (Theorem 8.1). The tops add up to top(u_m) = (4/3)m² + O(m).
Consider the steps taken after G first reaches X/2: there are at least (X/2 − m)/m of them, and each costs more than X/2.
So the final bound X satisfies X(X − 2m) ≤ (16/3)m³ + O(m²) (unless X ≤ 2G_0), i.e. X ≤ 2.31 m^{3/2}(1 + o(1)).
[Proved, under the stated two-sided hypothesis.] So no choice of ideals rescues the "every kernel element dies in the
quotient" mechanism.

### 8.4 What a working argument must control  [Proved reformulations; Verified data]
(i) *Degree-sensitive criterion.* For every n:
        π_{m−a}(ker^{(m)}_n) = 0   ⟺   ω_{K_a}·(u_m)_{D−n} ⊂ σ'u_m .
Proof: ker^{(m)}_n = ^⊥(σ'u_m) in degree top_m − n (Frobenius), J_a^⊥ = ω_{K_a}u_m (§7.3), and top_m − T_a(m) = D.
By the window lemma, k_m ≥ k_{m−a} + (m−a) iff this holds for all n < k_{m−a} + m − a. C_a is the case n = D, the "worst"
degree, and that is the case that fails. Define n*(m,a) := least n with π_{m−a}(ker^{(m)}_n) ≠ 0 (`taskH_lift_check.py`):
| (m,a) | (6,1) | (7,1) | (8,1) | (9,1) | (10,1) | (7,2) | (8,2) | (9,2) | (10,2) | (8,3),(9,3) |
| n* − k_{m−a} | 10 | 10 | 11 | 7 (η) | 1 (free step) | 16 | 21 | 27 | 22 | never (C3 holds) |
| needed: m − a | 5 | 6 | 7 | 8 | 9 | 5 | 6 | 7 | 8 | 5, 6 |
Above n* the image fills ker^{(m−a)}_n quickly (e.g. m = 7, a = 1: all of it from n = 32 on).
(ii) *Homological form.* **ker(R^{(m)} | (u_m)_n) ≅ Tor_1^{u(L_{≥m})}(F2, U/Uσ')_{n+3}**, so k_m + 3 is the lowest degree of
the restricted homology H_1(L^s_{≥m}; W) of the deep ideal L^s_{≥m} ⊂ n̂ with coefficients in W := U/Uσ'.
Proof: U is free as a left u(L_{≥m})-module (PBW). It is free as a right F2[σ']-module: it is torsion free by Theorem 7.1,
and a bounded-below graded torsion-free module over the graded PID F2[σ'] is free. So both ways of computing
F2 ⊗^L_{u(L_{≥m})} U ⊗^L_{F2[σ']} F2 collapse: one gives Tor^{F2[σ']}(u_m, F2), with Tor_1 = ker R^{(m)} shifted by 3, the
other gives Tor^{u(L_{≥m})}(F2, U/Uσ'). ∎ A quadratic bound is therefore a **vanishing theorem for H_1(L_{≥m}; W) in
degrees < c·m²**. Note h_W = (1 − z³)h_U = h(U/Uc_3), so W is a "deformation" of U/Uc_3, whose H_1 starts in degree O(m)
(6.7(a)). The vanishing must come from the non-leading terms of σ'.

### 8.5 Other ideas examined  [Heuristic unless stated]
* *Evaluation modules with truncation (idea 1).* If yσ' ∈ I_m, then Ψ_r(y)·Δ_r ∈ M(𝔞_K) with 𝔞_K = (t_1^K, …, t_r^K),
  K = m/3, and Δ_r = det Ψ_r(σ') (or any "denominator" taken from the minimal polynomial). The modules ⊗V*[t_s]/(t_s^K)
  are u_{3K}-modules, and the products of their matrix coefficients contain all of u_{3K−2}^*, so two levels are lost
  [Proved]. Hence a colon-ideal statement (𝔞_K : Δ_r) ⊂ 𝔞_K in t-degree < f(m)/3 would prove injectivity directly. It
  fails at t-degree K − 1 for K = 2^j and **every** r [Proved + computation: the pair cancellation is the identity
  Ψ_2(σ') = (t_1+t_2)I of `taskF_symb.py`].
  The reason is that Ψ_r(σ') = s_1·I + Σ_{triples} N_{abc}, where s_1 = Σ t_s. The pair terms are the mixed part of Ψ_2
  placed at two factors, so they cancel. The triple term is
  Σ_{π∈S_3} ρ*(e_{π0}) ⊗ ρ*(e_{π1}) ⊗ ρ*(e_{π2}) on the factors a, b, c (each cyclic word contributes all 6 assignments,
  and 3 ≡ 1). Since ρ*(e_0), ρ*(e_1), ρ*(e_2) are matrix units with distinct sources, N_{abc} kills every basis tensor
  whose indices at a, b, c are not a permutation of {0, 1, 2}, e.g. f_0^{⊗r}. So Ψ_r(σ')f_0^{⊗r} = s_1 f_0^{⊗r}, and
  s_1 | det Ψ_r(σ') (for r = 3 this is the factor s_1^{21} of 7.2: 21 = 27 − 6 non-permutation tensors). Now
  s_1·s_1^{K−1} = Σ t_s^K ∈ 𝔞_K (Frobenius in characteristic 2). Hence s_1^{K−1}f_0^{⊗r} is a nonzero kernel vector of σ'
  on ⊗V*[t_s]/(t_s^K) in t-degree K − 1 for every r. This is the linear barrier of 7.2 again, and padding with more
  factors cannot remove it.
* *Contractions + semicontinuity.* For a superadditive weight ν on letters, gr_ν u_m = u(L^ν) (contracted bracket), and
  k(σ') ≥ k(σ_0) for the leading form σ_0. If all four PBW terms of σ' stay leading, the brackets [e1,e0] → y2.1,
  [e2,e1] → y2.2 and [y1.0,y2.2] → y3.1 must survive (equal ν-weights) [Proved]. In the two contractions tried by hand
  that kill more brackets, σ_0 becomes a product of letters with a kernel in degree ≤ 2. If only y3.1 survives in
  [y1.0, y2.2], then σ_0 = y2.2·y1.0, and y2.2 itself is in the kernel. If the e_i commute, then σ_0 = e0e1e2, and e0 is
  in the kernel. No useful contraction was found.

### 8.6 Status after Task H
* Proved: Theorem 8.1 (C_a(m) ⟹ k_{m−a} ≤ T_a(m) − 3, with forced lifting near the top); C3 fails in every 61
  consecutive levels; no a(m) = o(√m) works; every bound from iterating 7.4 is ≤ b + 4m^{3/2} (Theorem 8.3); the same
  O(m^{3/2}) ceiling for general ideal chains (Remark 8.4, two-sided hypothesis); the Tor/H_1 reformulation 8.4(ii).
* Not proved: any superlinear lower bound for k_m. The best unconditional bound is still k_m ≥ max(82, m − 3) (m ≥ 14).
* [Conjecture] The degree-sensitive criterion 8.4(i) holds with large margin (n*(m,a) − k_{m−a} ≫ m − a at jump steps, as
  in the table). A proof of k_m/m^{3/2} → ∞ has to establish vanishing of π_{m−a} on the kernel only in a window above
  k_{m−a}, or equivalently vanishing of H_1(L_{≥m}; U/Uσ') below a superlinear degree. Statements that are uniform in the
  degree (C_a, annihilators, integrals) are provably insufficient.
