r"""
A SageMath solver for Thue-Mahler equations, following the algorithm of

    A. Gherga and S. Siksek, "Efficient resolution of Thue-Mahler equations",
    Algebra & Number Theory 19:4 (2025), 667-713.

We solve
    F(X, Y) = a * p1^z1 * ... * pv^zv,    X, Y in Z, gcd(X, Y) = gcd(a0, Y) = 1
where F is an irreducible binary form of degree d >= 3 with integer
coefficients, a is a nonzero integer and p1, ..., pv are distinct primes
(pi not dividing a).

The algorithm works only in the number field K = Q(theta) of degree d, where
theta is a root of the monic polynomial f(x) = a0^{d-1} F(x/a0, 1).
"""

from sage.all import (
    ZZ,
    QQ,
    RealField,
    ComplexField,
    PolynomialRing,
    NumberField,
    matrix,
    vector,
    gcd,
    lcm,
    inverse_mod,
    prod,
    log,
    exp,
    pi,
    sqrt,
    ceil,
    floor,
    Integer,
    next_prime,
)

from sage.modules.free_module_integer import IntegerLattice
from sage.rings.finite_rings.integer_mod_ring import IntegerModRing
from fpylll import IntegerMatrix as FPIntegerMatrix, LLL, CVP
from itertools import product


class ThueMahlerSolver:
    r"""
    Solve the Thue-Mahler equation  F(X,Y) = a * prod_{p in S} p^{z_p}.

    INPUT:

    - ``F`` -- a binary form: either a list/tuple of integer coefficients
      ``[a0, a1, ..., ad]`` with ``F = a0 X^d + a1 X^{d-1} Y + ... + ad Y^d``,
      a univariate polynomial (interpreted as ``F(X, 1)``), or a homogeneous
      polynomial in two variables.
    - ``S`` -- a list/set of distinct rational primes.
    - ``a`` -- (default 1) the fixed nonzero integer multiplier.
    """

    def __init__(self, F, S, a=1, prec=None, verbose=False):
        self.verbose = verbose
        self.a = ZZ(a)
        self.S = sorted(set(ZZ(p) for p in S))
        for p in self.S:
            assert p.is_prime(), "S must contain only primes"

        # Normalise F to a list of integer coefficients [a0, ..., ad].
        self.coeffs = self._normalise_form(F)
        self.d = len(self.coeffs) - 1
        assert self.d >= 3, "degree must be at least 3"
        self.a0 = self.coeffs[0]
        assert self.a0 != 0

        # Monic polynomial f(x) = x^d + a1 x^{d-1} + a0 a2 x^{d-2} + ...
        x = PolynomialRing(QQ, "x").gen()
        f = x ** self.d
        for i in range(1, self.d + 1):
            f += (self.a0 ** (i - 1)) * self.coeffs[i] * x ** (self.d - i)

        self.K = NumberField(f, "theta")
        self.theta = self.K.gen()
        self.f_poly = f

        # Precision for real/complex arithmetic.
        self.prec = prec if prec is not None else 5000
        self.RR = RealField(self.prec)
        self.CC = ComplexField(self.prec)

        # Field invariants.
        self.signature = self.K.signature()
        self.u, self.v = self.signature
        self._real_embs = None
        self._complex_embs = None
        self._Cl = None
        self._unit_group = None
        self._SU = None
        self._SU_primes = None
        self._D = None
        self._c17_cache = {}
        self._adequate_cache = {}

    # ------------------------------------------------------------------ #
    #  input normalisation
    # ------------------------------------------------------------------ #
    def _normalise_form(self, F):
        from sage.all import Polynomial, Parent
        if isinstance(F, (list, tuple)):
            return [ZZ(c) for c in F]
        try:
            is_poly = isinstance(F, Polynomial)
        except Exception:
            is_poly = False
        if not is_poly:
            raise TypeError("F must be a list of coefficients or a polynomial")
        R = F.parent()
        if R.ngens() == 1:
            # F(X, 1)
            c = F.list()
            while len(c) < 2:
                c = c + [0]
            return [ZZ(v) for v in c]
        if R.ngens() == 2:
            X, Y = R.gens()
            d = F.degree()
            return [ZZ(F.coefficient({X: d - i, Y: i})) for i in range(d + 1)]
        raise TypeError("F must be a binary form in one or two variables")

    # ------------------------------------------------------------------ #
    #  valuation / residue helpers
    # ------------------------------------------------------------------ #
    def ord_P(self, alpha, P):
        r"""Return ord_P(alpha) for alpha in K^*.  May be negative."""
        return self.K.ideal(alpha).valuation(P)

    def residue_of(self, alpha, P):
        r"""Image of alpha in the residue field F_P = O_K / P."""
        return P.residue_field()(alpha)

    def is_in_prime_subfield(self, alpha, P):
        r"""True iff the image of alpha in F_P lies in the prime subfield F_p."""
        r = self.residue_of(alpha, P)
        p = P.smallest_integer()
        return r ** p == r

    def residue_int(self, alpha, P):
        r"""
        Assume the image of alpha in F_P lies in the prime subfield F_p.
        Return the unique integer u in {0,...,p-1} with alpha = u (mod P).
        """
        r = self.residue_of(alpha, P)
        return ZZ(r)

    def theta_mod_pk(self, P, k):
        r"""
        Hensel lift: the unique integer theta0 in {0, ..., p^k - 1} with
        theta = theta0 (mod P^k), where P is a prime above p with e=f=1.
        """
        if k <= 0:
            return ZZ(0)
        p = ZZ(P.norm())
        r = self.residue_int(self.theta, P)  # root of f mod p
        f = self.f_poly
        x = f.parent().gen()
        df = f.derivative()
        fp_val = ZZ(df(r) % p)
        inv_fp = inverse_mod(fp_val, p)
        pk = p ** k
        for i in range(1, k):
            pi_i = p ** i
            c = (f(r) // pi_i) % p  # f(r) = c * p^i (mod p^{i+1})
            t = (-c * inv_fp) % p
            r = r + t * pi_i
            r %= pk
        return r

    # ------------------------------------------------------------------ #
    #  Section 2: adequate / satisfactory sets (Algorithm 2.6)
    # ------------------------------------------------------------------ #
    def _adequate(self, p, alpha, beta, depth=0):
        r"""
        Compute adequate sets (L, M) for the pair (alpha, beta):
        for every U in Z_(p), either the p-part of beta (U + alpha) O_K is in L,
        or it equals b * P^l for some (b, P) in M and l >= 0.

        ``alpha`` must satisfy K = Q(alpha).
        """
        Pv = self.K.ideal(p).prime_factors()

        B = []
        for P in Pv:
            if self.ord_P(alpha, P) >= 0 and self.is_in_prime_subfield(alpha, P):
                B.append(P)

        # b = prod_{P|p} P^{ord_P(beta) + min{0, ord_P(alpha)}}
        b = self.K.ideal(1)
        for P in Pv:
            e = self.ord_P(beta, P) + min(0, self.ord_P(alpha, P))
            b = b * P ** e

        if len(B) == 0:
            return [b], []

        if len(B) == 1 and B[0].ramification_index() == 1 and B[0].residue_class_degree() == 1:
            return [], [(b, B[0])]

        # U = {u in 0..p-1 : alpha = -u mod P for some P in B}
        Uset = set()
        for P in B:
            u = self.residue_int(-alpha, P)
            Uset.add(u)

        Lall = []
        Mall = []
        for u in Uset:
            a2 = (u + alpha) / p
            b2 = p * beta
            Lu, Mu = self._adequate(p, a2, b2, depth + 1)
            Lall.extend(Lu)
            Mall.extend(Mu)

        if set(Uset) == set(range(p)):
            return Lall, Mall
        else:
            return [b] + Lall, Mall

    def satisfactory_sets(self, p):
        r"""Return the satisfactory pair (L_p, M_p) for the prime p."""
        alpha = -self.theta / self.a0
        beta = self.a0
        L, M = self._adequate(p, alpha, beta)
        Lp = list(L) + [self.K.ideal(1)]
        Mp = list(M)
        # Refinements (optional but cheap).
        Lp, Mp = self._refine_satisfactory(p, Lp, Mp)
        return Lp, Mp

    def _refine_satisfactory(self, p, Lp, Mp):
        # (i) replace (b, P) by (b / P^{ord_P(b)}, P) : the P-power is absorbed
        #     into the unknown exponent l.
        Mp = [(b / P ** self.ord_P(b, P), P) for (b, P) in Mp]
        # (ii) delete b in Lp if some (b', P) in Mp has b' | b and
        #      b/b' = (p O_K)^w for some w >= 0.
        newL = []
        for b in Lp:
            dominated = False
            for (bp, P) in Mp:
                q = b / bp
                if q.is_integral() and self._ideal_is_pO_power(q, p):
                    dominated = True
                    break
            if not dominated:
                newL.append(b)
        Lp = newL
        # (iii) if p | a0 then delete b with ord_p(Norm(b)) < (d-1) ord_p(a0).
        if self.a0 % p == 0:
            threshold = (self.d - 1) * self.ord_p_int(self.a0, p)
            Lp = [b for b in Lp if self._norm_ordp(b) >= threshold]
        return Lp, Mp

    def _ideal_is_pO_power(self, ideal, p):
        r"""True iff ``ideal`` equals (p O_K)^w for some integer w >= 0."""
        if ideal == self.K.ideal(1):
            return True
        pO = self.K.ideal(p)
        w = 1
        while True:
            q = pO ** w
            if ideal == q:
                return True
            # q is integral and its support is on primes above p; once its norm
            # exceeds the norm of ideal, no larger power can match.
            if ZZ(q.norm()) > ZZ(ideal.norm()):
                return False
            w += 1

    def _norm_ordp(self, ideal):
        p = self._p_of_ideal(ideal)
        if p == 1:
            return 0
        return self.ord_p_int(ZZ(ideal.norm()), p)

    def _p_of_ideal(self, ideal):
        # the unique rational prime below the (p-power) ideal; 1 if empty
        if ideal == self.K.ideal(1):
            return ZZ(1)
        fac = ideal.factor()
        if len(fac) == 0:
            return ZZ(1)
        return fac[0][0].smallest_integer()

    @staticmethod
    def ord_p_int(n, p):
        r"""p-adic valuation of the rational integer n."""
        n = ZZ(n)
        if n == 0:
            raise ValueError("valuation of 0")
        v = 0
        while n % p == 0:
            v += 1
            n //= p
        return v

    # ------------------------------------------------------------------ #
    #  ideals of a given norm
    # ------------------------------------------------------------------ #
    def _ideals_of_prime_power_norm(self, p, e):
        primes = self.K.ideal(p).prime_factors()
        result = []

        def rec(idx, remaining, current):
            if idx == len(primes):
                if remaining == 0:
                    I = self.K.ideal(1)
                    for P, ep in zip(primes, current):
                        if ep:
                            I = I * P ** ep
                    result.append(I)
                return
            fP = primes[idx].residue_class_degree()
            ep = 0
            while ep * fP <= remaining:
                rec(idx + 1, remaining - ep * fP, current + [ep])
                ep += 1

        rec(0, e, [])
        return result

    def _ideals_of_norm(self, R):
        R = ZZ(R)
        if R == 1:
            return [self.K.ideal(1)]
        result = [self.K.ideal(1)]
        for p, e in R.factor():
            part = self._ideals_of_prime_power_norm(p, e)
            result = [I * J for I in result for J in part]
        return result

    # ------------------------------------------------------------------ #
    #  Section 3: ideal equations  (a0 X - theta Y) O_K = a * p1^n1 ...
    # ------------------------------------------------------------------ #
    def _ideal_equations(self):
        data = []
        for p in self.S:
            Lp, Mp = self.satisfactory_sets(p)
            data.append((p, Lp, Mp))

        # Each prime contributes either an ideal from Lp (fixed p-part) or a
        # pair (b, P) from Mp (fixed part b, unknown exponent prime P).
        combos = [(self.K.ideal(1), [])]
        for p, Lp, Mp in data:
            new_combos = []
            for b_prod, S_list in combos:
                for b in Lp:
                    new_combos.append((b_prod * b, list(S_list)))
                for b, P in Mp:
                    new_combos.append((b_prod * b, S_list + [P]))
            combos = new_combos

        A = abs(self.a) * abs(self.a0) ** (self.d - 1)
        Z = []
        for b_prod, S_list in combos:
            nb = ZZ(b_prod.norm())
            R = A // gcd(nb, A)
            for b0 in self._ideals_of_norm(R):
                if self._b0_allowed(b0, R):
                    a_ideal = b0 * b_prod
                    Z.append((a_ideal, tuple(S_list)))
        return Z

    def _p_part(self, ideal, p):
        r"""The p-part of ``ideal`` : prod_{P|p} P^{ord_P(ideal)}."""
        result = self.K.ideal(1)
        for P in self.K.ideal(p).prime_factors():
            e = ideal.valuation(P)
            result = result * P ** e
        return result

    def _b0_allowed(self, b0, R):
        r"""
        Refinement from the remark after Proposition 3.1: for each prime p | R
        the p-part of b0 must be realisable from the satisfactory sets of p.
        """
        for p, _ in R.factor():
            Lp, Mp = self.satisfactory_sets(p)
            b0_p = self._p_part(b0, p)
            if b0_p in Lp:
                continue
            ok = False
            for b, P in Mp:
                q = b0_p / b
                if q.is_integral():
                    fac = q.factor()
                    if all(Q == P for Q, _e in fac):
                        ok = True
                        break
            if not ok:
                return False
        return True

    # ------------------------------------------------------------------ #
    #  Section 4: making the ideals principal
    # ------------------------------------------------------------------ #
    def _class_group(self):
        if self._Cl is None:
            self._Cl = self.K.class_group()
        return self._Cl

    def _solve_congruences(self, A, t, mods):
        r"""
        Solve  sum_i r[i] * A[i] = t  (mod mods componentwise).

        A is a list of s integer vectors of length k, t a length-k vector,
        mods a length-k tuple of positive integers.  Returns r (length s) or
        raises ValueError if no solution exists.
        """
        s = len(A)
        k = len(mods)
        if k == 0:
            return [0] * s
        if s == 0:
            if all(t[j] % mods[j] == 0 for j in range(k)):
                return []
            raise ValueError("no solution")
        # coefficient matrix C (k rows, s+k columns) for C * [r; m] = t
        C = matrix(ZZ, k, s + k)
        for j in range(k):
            for i in range(s):
                C[j, i] = A[i][j]
            C[j, s + j] = -mods[j]
        tv = vector(ZZ, [ZZ(t[j]) for j in range(k)])
        D, U, V = C.smith_form()
        # D = U * C * V,  so C = U^{-1} D V^{-1}.
        t2 = U * tv
        yprime = [ZZ(0)] * (s + k)
        for j in range(k):
            d = D[j, j]
            if d == 0:
                if t2[j] != 0:
                    raise ValueError("no solution")
            else:
                if t2[j] % d != 0:
                    raise ValueError("no solution")
                yprime[j] = t2[j] // d
        y = V * vector(ZZ, yprime)
        return [ZZ(y[i]) for i in range(s)]

    def _principalize(self, a_ideal, S_ideals):
        r"""
        Given an ideal equation (a_ideal, S_ideals), return the list of tuples
        (tau, deltas) where deltas = [d1, ..., dr] is the S-unit basis and
        tau = zeta^a * alpha.  Returns [] if the class-group obstruction makes
        the equation impossible.
        """
        Cl = self._class_group()
        invariants = tuple(Cl.invariants())
        k = len(invariants)
        s = len(S_ideals)

        A = [list(Cl(P).exponents()) for P in S_ideals]
        c_exps = list(Cl(a_ideal).exponents())
        t = [(-ZZ(e)) % n for e, n in zip(c_exps, invariants)]

        try:
            r = self._solve_congruences(A, t, invariants)
        except ValueError:
            return []

        I = a_ideal
        for P, ri in zip(S_ideals, r):
            I = I * P ** ri
        if not I.is_principal():
            raise RuntimeError("internal error: ideal should be principal")
        alpha = I.gens_reduced()[0]

        zeta = self._sunit_generator_value(S_ideals)
        m = self.K.zeta_order()
        deltas = self._sunit_basis(S_ideals)

        out = []
        for a in range(m // 2):
            tau = zeta ** a * alpha
            out.append((tau, deltas))
        return out

    def _sunit_group(self, S_ideals):
        key = tuple(S_ideals)
        if self._SU is None or self._SU_primes != key:
            self._SU = self.K.S_unit_group(S=list(S_ideals))
            self._SU_primes = key
        return self._SU

    def _sunit_generator_value(self, S_ideals):
        SU = self._sunit_group(S_ideals)
        return SU.gens_values()[0]

    def _sunit_basis(self, S_ideals):
        SU = self._sunit_group(S_ideals)
        return list(SU.gens_values())[1:]

    def reduce_to_s_unit_equations(self):
        r"""
        Run Sections 2-4: return the list of S-unit equations
            a0 X - theta Y = tau * d1^b1 * ... * dr^br
        as a list of tuples (tau, deltas).
        """
        Z = self._ideal_equations()
        result = []
        for a_ideal, S_ideals in Z:
            result.extend(self._principalize(a_ideal, S_ideals))
        return result

    # ------------------------------------------------------------------ #
    #  embeddings and heights
    # ------------------------------------------------------------------ #
    def _embeddings(self):
        r"""
        Return (real_embs, complex_embs) where real_embs are the real
        embeddings and complex_embs is one representative of each pair of
        complex conjugate embeddings (a total of u + v embeddings).
        """
        if self._real_embs is None or self._complex_embs is None:
            real = self.K.real_embeddings(self.prec)
            ce = self.K.complex_embeddings(self.prec)
            complex_all = [e for e in ce if abs(e(self.theta).imag()) > 1e-30]
            seen = []
            reps = []
            for e in complex_all:
                t = e(self.theta)
                key = t if t.imag() > 0 else t.conjugate()
                if not any((key - s).abs() < 1e-50 for s in seen):
                    seen.append(key)
                    reps.append(e)
            self._real_embs = real
            self._complex_embs = reps
        return self._real_embs, self._complex_embs

    def _height(self, alpha):
        return self.RR(alpha.global_height())

    # ------------------------------------------------------------------ #
    #  Section 6: initial bound on B (Matveev + Yu)
    # ------------------------------------------------------------------ #
    def _compute_D(self):
        r"""
        Degree of the minimal extension Q(theta1, theta2, theta3) generated
        by three conjugates of theta, deduced from the Galois group of f.
        """
        if self._D is not None:
            return self._D
        f = self.f_poly
        G = f.galois_group()
        try:
            P = G.as_permutation_group()
        except Exception:
            P = G
        order = ZZ(P.order())
        d = self.d
        best = None
        for i in range(1, d + 1):
            Si = P.stabilizer(i)
            for j in range(i + 1, d + 1):
                Sij = Si.stabilizer(j)
                for k in range(j + 1, d + 1):
                    Sijk = Sij.stabilizer(k)
                    idx = order // ZZ(Sijk.order())
                    if best is None or idx < best:
                        best = idx
        self._D = best
        return best

    def _initial_bound(self, tau, deltas, S_ideals):
        r"""
        Compute the initial upper bound B <= c20 (Proposition 6.8) for the
        S-unit equation  a0 X - theta Y = tau * d1^b1 * ... * dr^br.
        """
        RR = self.RR
        d = self.d
        r = len(deltas)
        s = len(S_ideals)
        u, v = self.signature
        e = RR(exp(RR(1)))

        D = RR(self._compute_D())

        htheta = self._height(self.theta)
        htau = self._height(tau)
        hdelta = [self._height(dj) for dj in deltas]

        c7 = RR(log(2)) + 2 * htheta + htau
        c8 = 2 * D * c7

        # Lemma 6.2 (Matveev, infinite places)
        hstar = [sqrt(4 * hdelta[j] ** 2 + (RR(pi) ** 2) / (D ** 2)) for j in range(r)]
        hstar_r1 = sqrt(4 * c7 ** 2 + (RR(pi) ** 2) / (D ** 2))
        c9 = RR(6) * (RR(30) ** (r + 5)) * (RR(r + 2) ** RR(5.5)) * (D ** (r + 3)) \
            * RR(log(e * D)) * prod(hstar) * hstar_r1
        c10 = c8 + c9 * RR(log(e * (r + 1)))

        # Lemma 6.3 (Yu, finite places)
        hdagger = [RR(max(2 * hdelta[j], RR(1) / (16 * e ** 2 * D ** 2))) for j in range(r)]
        hdagger_r1 = RR(max(2 * c7, RR(1) / (16 * e ** 2 * D ** 2)))
        c1 = (16 * e * D) ** (2 * (r + 1) + 2) * RR(r + 1) ** RR(5.0 / 2) \
            * RR(log(2 * (r + 1) * D)) * RR(log(2 * D))
        T = set(ZZ(P.smallest_integer()) for P in S_ideals)
        c11 = self._yu_c11(T, D, r)
        c12 = c1 * c11 * prod(hdagger) * hdagger_r1

        # Lemma 6.4
        num_MKinf = u + v
        c13 = RR(num_MKinf + 2 * s) / RR(d)
        c14 = 2 * htau + c13 * RR(max(c8, c10))
        c15 = c13 * RR(max(c9, c12))

        # Lemma 6.6
        c16 = c14 + 2 * RR(log(2)) + 2 * htheta + htau

        # Lemma 6.7 : c17 from the regulator matrix
        c17 = self._c17(deltas, S_ideals)

        # Proposition 6.8
        c18 = 2 * d * c17 * c16
        c19 = 2 * d * c17 * c15
        c20 = 2 * c18 + RR(max(2 * c19 * RR(log(c19)), 4 * e ** 2))
        return RR(c20)

    def _yu_c11(self, T, D, r):
        RR = self.RR
        d = self.d
        # c2(n,P) = u p^v / ((r+1) v log p) <= (D/d) p^(D/d) / log p (safe upper
        # bound), with uv <= D/d, u,v >= 1.
        vals = []
        for p in T:
            p = RR(p)
            vals.append((RR(D) / RR(d)) * (p ** (D / d)) / RR(log(p)))
        if not vals:
            return RR(1)
        return RR(max(vals))

    def _place_log_norm(self, delta, place, is_complex=False):
        r"""log ||delta||_place, for place a prime ideal or an embedding."""
        RR = self.RR
        if hasattr(place, "norm"):
            return -ZZ(delta.valuation(place)) * RR(log(place.norm()))
        val = place(delta)
        if is_complex:
            return 2 * RR(log(abs(val)))
        return RR(log(abs(val)))

    def _c17(self, deltas, S_ideals):
        r"""
        Compute c17 from Lemma 6.7: iterate over the r+1 possible sets U and
        take the smallest max-absolute-entry of the inverse regulator matrix.
        """
        key = (tuple(deltas), tuple(S_ideals))
        if key in self._c17_cache:
            return self._c17_cache[key]
        RR = self.RR
        real_embs, complex_embs = self._embeddings()
        # places with a flag indicating complex embeddings
        places = ([(P, False) for P in S_ideals]
                  + [(e, False) for e in real_embs]
                  + [(e, True) for e in complex_embs])
        r = len(deltas)
        best = None
        for skip in range(len(places)):
            U = [places[i] for i in range(len(places)) if i != skip]
            if len(U) != r:
                continue
            M = matrix(RR, r, r)
            for row, (place, is_c) in enumerate(U):
                for col, dj in enumerate(deltas):
                    M[row, col] = self._place_log_norm(dj, place, is_c)
            if M.rank() < r:
                continue
            Minv = M.inverse()
            cmax = max(abs(Minv[i, j]) for i in range(r) for j in range(r))
            if best is None or cmax < best:
                best = cmax
        best = RR(best)
        self._c17_cache[key] = best
        return best

    # ------------------------------------------------------------------ #
    #  lattice helpers (CVP)
    # ------------------------------------------------------------------ #
    def _lattice_basis(self, generators):
        r"""Basis (list of ZZ vectors) of the lattice spanned by ``generators``."""
        M = matrix(ZZ, [list(g) for g in generators])
        L = IntegerLattice(M)
        return [tuple(row) for row in L.basis_matrix().rows()]

    def _cvp_distance(self, basis, w):
        r"""Return D(L, w) = min_{x in L} ||w - x||_2 for a full-rank lattice L."""
        n = len(w)
        Bm = FPIntegerMatrix.from_matrix(matrix(ZZ, [list(b) for b in basis]))
        LLL.reduction(Bm)
        target = tuple(-ZZ(c) for c in w)
        cv = CVP.closest_vector(Bm, target)
        d2 = sum((ZZ(w[i]) + ZZ(cv[i])) ** 2 for i in range(n))
        return sqrt(self.RR(d2))

    def _babai(self, basis_rows, w):
        r"""
        Babai's nearest-plane: return (coefficient_vector, exact squared
        distance) of an approximate closest lattice vector to w.
        """
        RR = self.RR
        n = len(w)
        b = [vector(RR, [RR(x) for x in row]) for row in basis_rows]
        t = vector(RR, [RR(x) for x in w])
        bstar = [None] * n
        mu = [[RR(0)] * n for _ in range(n)]
        for i in range(n):
            bs = b[i]
            for j in range(i):
                mu[i][j] = b[i].dot_product(bstar[j]) / bstar[j].dot_product(bstar[j])
                bs -= mu[i][j] * bstar[j]
            bstar[i] = bs
        ccoef = [ZZ(0)] * n
        for i in range(n - 1, -1, -1):
            s = t
            for j in range(i + 1, n):
                s -= RR(ccoef[j]) * b[j]
            ccoef[i] = self._nearest_int(s.dot_product(bstar[i]) / bstar[i].dot_product(bstar[i]))
        v = [ZZ(0)] * n
        for i in range(n):
            if ccoef[i] != 0:
                for k in range(n):
                    v[k] += ccoef[i] * ZZ(basis_rows[i][k])
        R = sum((ZZ(w[k]) - v[k]) ** 2 for k in range(n))
        return ccoef, R

    def _closest_distance(self, basis_rows, w):
        r"""
        Exact D(L, w) = min_{x in L} ||w - x||_2 for a full-rank lattice L
        (uses Fincke-Pohst with high precision; avoids fpylll's double CVP).
        """
        RR = self.RR
        n = len(w)
        M = matrix(ZZ, [list(b) for b in basis_rows])
        Mred = M.LLL()
        basis_rows = [tuple(row) for row in Mred.rows()]
        B = matrix(ZZ, [list(b) for b in basis_rows])
        G = B * B.transpose()
        g = -(B * vector(ZZ, list(w)))  # q(x) = x^T G x + 2 g^T x + c = ||w - B^T x||^2
        c = sum(ZZ(x) ** 2 for x in w)
        ccoef, R = self._babai(basis_rows, w)
        sols = self._enum_quadratic(G, g, c, int(R))
        minv = R
        for x in sols:
            xv = vector(ZZ, x)
            q = xv.dot_product(G * xv) + 2 * g.dot_product(xv) + c
            if q < minv:
                minv = q
        return sqrt(RR(minv))

    def _shortest_nonzero(self, basis_rows):
        r"""Length of the shortest nonzero vector of the lattice spanned by rows."""
        RR = self.RR
        M = matrix(ZZ, [list(b) for b in basis_rows])
        Mred = M.LLL()
        basis_rows = [tuple(row) for row in Mred.rows()]
        n = len(basis_rows)
        B = matrix(ZZ, [list(b) for b in basis_rows])
        G = B * B.transpose()
        g = vector(ZZ, [ZZ(0)] * n)
        c = ZZ(0)
        bound = min(sum(ZZ(x) ** 2 for x in row) for row in basis_rows)
        sols = self._enum_quadratic(G, g, c, int(bound))
        minv = None
        for x in sols:
            if all(ZZ(xi) == 0 for xi in x):
                continue
            xv = vector(ZZ, x)
            q = xv.dot_product(G * xv)
            if minv is None or q < minv:
                minv = q
        if minv is None:
            minv = bound
        return sqrt(RR(minv))

    # ------------------------------------------------------------------ #
    #  Section 7: controlling ord_p(a0 X - theta Y)  (Proposition 7.2)
    # ------------------------------------------------------------------ #
    def _split_fractional(self, ideal):
        r"""Split a fractional ideal into (T1, T2) coprime integral ideals."""
        T1 = self.K.ideal(1)
        T2 = self.K.ideal(1)
        for P, e in ideal.factor():
            if e > 0:
                T1 = T1 * P ** e
            elif e < 0:
                T2 = T2 * P ** (-e)
        return T1, T2

    def _prop72(self, p, k, tau, deltas, B2):
        r"""
        Proposition 7.2: try to prove  ord_p(a0 X - theta Y) <= k - 1.
        Returns True if established, False otherwise.
        """
        p_rat = ZZ(p.smallest_integer())
        theta0 = self.theta_mod_pk(p, k)
        a = self.K.ideal(p_rat) / p

        T1, T2 = self._split_fractional(self.K.ideal(tau))
        g1 = (a ** k) + self.K.ideal(self.theta - theta0)
        g2 = (a ** k) + T1
        if g1 != g2:
            return True

        b = a ** k / g2
        ratios = []
        for Q, e in b.factor():
            ratios.append(QQ(ZZ(e)) / ZZ(Q.ramification_index()))
        kprime = ceil(max(ratios))

        G = b.idealstar(flag=2)
        inv = tuple(ZZ(c) for c in G.invariants())
        bid = b._pari_bid_(2)
        nf = self.K.pari_nf()

        def dlog(elt):
            return [ZZ(c) for c in nf.ideallog(elt.__pari__(), bid)]

        # H = image of (Z / p^{kprime} Z)^x in G
        hcols = []
        for g in IntegerModRing(p_rat ** kprime).unit_gens():
            hcols.append(dlog(self.K(g)))

        Acols = [dlog(dj) for dj in deltas]
        tau0 = (theta0 - self.theta) / tau
        t = dlog(tau0)

        # combined columns: C * [n ; m] = A n - H m
        Ccols = Acols + [[-c for c in h] for h in hcols]
        try:
            sol = self._solve_congruences(Ccols, t, inv)
        except ValueError:
            return True
        w = sol[:len(deltas)]

        # L = {n : exists m, A n - H m = 0} ; D(L, w)
        D = self._kernel_coset_distance(Ccols, inv, len(deltas), w)
        return D > B2

    def _kernel_coset_distance(self, columns, invariants, r, w):
        r"""D(L, w) where L = {n in Z^r : exists m, C[n;m] == 0 (mod inv)}."""
        g = len(invariants)
        m = len(columns)
        C = matrix(ZZ, g, m)
        for j in range(m):
            for i in range(g):
                C[i, j] = columns[j][i]
        B = matrix(ZZ, g, m + g)
        for i in range(g):
            for j in range(m):
                B[i, j] = C[i, j]
            B[i, m + i] = -invariants[i]
        K = B.right_kernel()
        gens = [list(v[:r]) for v in K.basis()]
        basis = self._lattice_basis(gens)
        if len(basis) != r or matrix(ZZ, [list(b) for b in basis]).rank() < r:
            # kernel projection not full rank: treat conservatively
            return self.RR(0)
        return self._cvp_distance(basis, w)

    # ------------------------------------------------------------------ #
    #  Section 7 (valuation bounds) and Section 8.1 (updating B1, B2)
    # ------------------------------------------------------------------ #
    def _valuation_bounds(self, tau, deltas, S_ideals, B2):
        r"""Return k_j (the maximal ord_{p_j}(a0 X - theta Y)) for each p_j in S."""
        RR = self.RR
        d = self.d
        r = len(deltas)
        k_bounds = []
        for p in S_ideals:
            p_rat = ZZ(p.smallest_integer())
            k = max(1, int(ceil(r * RR(log(B2)) / ((d - 2) * RR(log(p_rat))))))
            while True:
                if self._prop72(p, k, tau, deltas, B2):
                    k_bounds.append(k - 1)
                    break
                k += 1
                if k > 200000:
                    k_bounds.append(k - 1)
                    break
        return k_bounds

    def _update_B1B2(self, tau, deltas, S_ideals, k_bounds, Binf):
        r"""Section 8.1: update the 1-norm and 2-norm bounds."""
        RR = self.RR
        u, v = self.signature
        s = len(S_ideals)
        nunits = u + v - 1
        kp = [self.ord_P(tau, p) for p in S_ideals]
        kpp = [k_bounds[j] - kp[j] for j in range(s)]
        if s > 0:
            M0 = matrix(RR, s, s)
            for j in range(s):
                for i in range(s):
                    dj = deltas[nunits + i]
                    M0[j, i] = -ZZ(dj.valuation(S_ideals[j])) * RR(log(S_ideals[j].norm()))
            Minv = M0.inverse()
        else:
            Minv = matrix(RR, 0, 0)
        rho = []
        for i in range(s):
            rho_i = RR(0)
            for j in range(s):
                rho_i += abs(Minv[i, j]) * RR(log(S_ideals[j].norm())) * RR(max(abs(kp[j]), abs(kpp[j])))
            rho_i = min(RR(Binf), rho_i)
            rho.append(rho_i)
        B1 = RR(nunits) * RR(Binf) + sum(rho, RR(0))
        B2 = sqrt(RR(nunits) * RR(Binf) ** 2 + sum([ri ** 2 for ri in rho], RR(0)))
        return B1, B2

    @staticmethod
    def _nearest_int(x):
        return int(floor(x + QQ(1) / QQ(2)))

    # ------------------------------------------------------------------ #
    #  Sections 8-9: reduction of the bound B via an approximation lattice
    # ------------------------------------------------------------------ #
    def _reduce_bound_once(self, tau, deltas, S_ideals, Binf):
        r"""
        One reduction iteration.  Returns (Binf, B1, B2, k_bounds) with Binf
        possibly improved.
        """
        RR = self.RR
        r = len(deltas)
        u, v = self.signature
        d = self.d
        s = len(S_ideals)
        nunits = u + v - 1

        c17 = self._c17(deltas, S_ideals)
        B1 = RR(r) * RR(Binf)
        B2 = sqrt(RR(r)) * RR(Binf)

        k_bounds = self._valuation_bounds(tau, deltas, S_ideals, B2)
        B1, B2 = self._update_B1B2(tau, deltas, S_ideals, k_bounds, Binf)

        if u == 0:
            # totally imaginary: Proposition 8.3 gives a final bound directly
            c21 = self._c21(tau, S_ideals, k_bounds)
            c22 = self._c22(tau)
            newB = 2 * c17 * (c21 + c22)
            return newB, B1, B2, k_bounds

        # nontotally complex: iterate over the real embedding sigma
        real_embs, complex_embs = self._embeddings()
        best_B = Binf
        for sigma in real_embs:
            nb = self._reduce_for_embedding(sigma, tau, deltas, S_ideals,
                                            k_bounds, Binf, B1, B2, c17)
            if nb is not None and nb < best_B:
                best_B = nb
        return best_B, B1, B2, k_bounds

    def _c21(self, tau, S_ideals, k_bounds):
        RR = self.RR
        total = RR(0)
        for j, p in enumerate(S_ideals):
            kp = self.ord_P(tau, p)
            kpp = k_bounds[j] - kp
            total += RR(max(0, kpp)) * RR(log(p.norm()))
        return total

    def _c22(self, tau):
        RR = self.RR
        real_embs, complex_embs = self._embeddings()
        total = RR(0)
        for sig in complex_embs:
            val = RR(abs(sig(tau))) / RR(abs(sig(self.theta).imag()))
            total += 2 * RR(log(max(RR(1), val)))
        return total

    def _reduce_for_embedding(self, sigma, tau, deltas, S_ideals, k_bounds,
                              Binf, B1, B2, c17):
        RR = self.RR
        r = len(deltas)
        u, v = self.signature
        d = self.d
        s = len(S_ideals)
        nunits = u + v - 1
        w = u + v - 2
        n = r + v

        real_embs, complex_embs = self._embeddings()
        ordered = [sigma]
        for e in real_embs:
            if not (e(self.theta) - sigma(self.theta)).abs() < 1e-50:
                ordered.append(e)
        for e in complex_embs:
            ordered.append(e)
        # ordered has u+v embeddings, ordered[0] = sigma = sigma_1
        thetas = [e(self.theta) for e in ordered]
        taus = [e(tau) for e in ordered]
        deltas_ij = [[e(dj) for e in ordered] for dj in deltas]

        # constants
        c21 = self._c21(tau, S_ideals, k_bounds)
        c22 = self._c22(tau)
        if u == 1:
            c23 = RR(1)
        else:
            c23 = None
            for a in range(u):
                for b in range(a + 1, u):
                    val = RR(abs(thetas[a] - thetas[b])) / (RR(abs(taus[a])) + RR(abs(taus[b])))
                    if c23 is None or val < c23:
                        c23 = val
        c24 = c21 + c22 + (u - 1) * RR(log(max(RR(1), RR(1) / c23)))
        c25 = exp(c24)
        c26 = RR(1) / (2 * c17)

        # c29(j) for j = 2..u+v  (index j is 1-based)
        c29 = {}
        for j in range(2, u + 1):  # 2..u (real, j>=2)
            c29[j] = RR(abs(taus[0])) * c25 / (RR(abs(taus[j - 1])) * c23)
        for j in range(u + 1, u + v + 1):  # complex
            c29[j] = RR(abs(taus[0])) * c25 / RR(abs(thetas[j - 1].imag()))

        # c30
        c30 = 2 * c17 * c24
        c28 = RR(abs(taus[0])) * c25 / min(RR(abs(thetas[j - 1].imag())) for j in range(u + 1, u + v + 1))
        c30 = RR(max(c30, RR(log(2 * c28)) / c26))
        if u >= 2:
            c27 = RR(abs(taus[0])) * c25 / (min(RR(abs(taus[j - 1])) for j in range(2, u + 1)) * c23)
            c30 = RR(max(c30, RR(log(2 * c27)) / c26))

        # approximate relations (d - 2 of them), stored as (beta, alpha list of length r, pi_index or None)
        rels = []
        for j in range(1, w + 1):  # Lemma 8.7
            beta = RR(log(abs((thetas[0] - thetas[1]) * taus[j + 1] / ((thetas[0] - thetas[j + 1]) * taus[1]))))
            alphas = [RR(log(abs(deltas_ij[i][j + 1] / deltas_ij[i][1]))) for i in range(r)]
            rels.append((beta, alphas, None))
        for j in range(1, v + 1):  # Lemma 8.8
            idx = u + j
            beta = RR(((thetas[0] - thetas[idx - 1]) / taus[idx - 1]).log().imag_part())
            alphas = [-RR(deltas_ij[i][idx - 1].log().imag_part()) for i in range(r)]
            rels.append((beta, alphas, j - 1))  # pi_index for b_{r+j}

        # A1, A2, B3, B4, B5
        A1 = (RR(1) + B1) / 2
        A2 = (2 * RR(pi) * (RR(1) + B1) + 1) / (2 * RR(pi))
        B3 = RR(0)
        for j in range(1, w + 1):
            B3 += (c29[2] + c29[j + 2]) ** 2
        for j in range(1, v + 1):
            B3 += c29[u + j] ** 2
        B3 = sqrt(B3)
        B4 = RR(0)
        for j in range(1, w + 1):
            B4 += A1 * (c29[2] + c29[j + 2])
        for j in range(1, v + 1):
            B4 += A2 * c29[u + j]
        B5 = sqrt(B2 ** 2 - w * Binf ** 2 + w * A1 ** 2 + v * A2 ** 2)

        # choose C; increase it until D > B5 (or give up)
        C0 = max(1, int(ceil(RR(B5) ** (n / max(1, (d - 2))))))
        D = None
        D2 = None
        C = None
        for trial in range(30):
            C = C0 * (2 ** trial)
            # build matrix M (n x n)
            M = matrix(ZZ, n, n)
            for k in range(s + 1):
                M[nunits + k, k] = 1
            for t in range(d - 2):
                col = s + 1 + t
                rel = rels[t]
                beta, alphas, pi_idx = rel
                for i in range(r):
                    M[i, col] = self._nearest_int(C * alphas[i])
                if pi_idx is not None:
                    M[r + pi_idx, col] = self._nearest_int(C * RR(pi))
            wvec = [0] * n
            for t in range(d - 2):
                wvec[s + 1 + t] = self._nearest_int(C * rels[t][0])

            Mmat = matrix(ZZ, [[M[i, j] for j in range(n)] for i in range(n)])
            Lb = [tuple(row) for row in Mmat.rows()]
            w_in_L = self._solve_linear_system(Mmat.transpose(), wvec) is not None
            if w_in_L:
                D = self._shortest_nonzero(Lb)
            else:
                D = self._closest_distance(Lb, wvec)
            D2 = D ** 2
            if D > B5:
                break
        if D is None or D <= B5:
            return None

        # numerically stable denominator of (64):  sqrt(a + B4^2) - B4
        a = RR(B3) * (D2 - B5 ** 2)
        denom = a / (sqrt(a + B4 ** 2) + B4)
        if denom <= 0:
            return None
        newB = RR(max(c30, (1 / c26) * RR(log(2 * C * B3 / denom))))
        return newB

    # ------------------------------------------------------------------ #
    #  Section 10: sieving and final enumeration
    # ------------------------------------------------------------------ #
    def _solve_linear_system(self, N, c):
        r"""Solve N x = c over Z (Smith form).  Returns a list or None."""
        s, r = N.dimensions()
        if s == 0:
            return [ZZ(0)] * r
        D, U, V = N.smith_form()
        c2 = U * vector(ZZ, [ZZ(x) for x in c])
        y = [ZZ(0)] * r
        for i in range(s):
            d = D[i, i] if i < r else ZZ(0)
            if d == 0:
                if c2[i] != 0:
                    return None
            else:
                if c2[i] % d != 0:
                    return None
                y[i] = c2[i] // d
        x = V * vector(ZZ, y)
        return [ZZ(x[i]) for i in range(r)]

    def _enum_quadratic(self, G, g, c, bound):
        r"""
        Enumerate all x in Z^m with  x^T G x + 2 g^T x + c <= bound,
        by the Fincke-Pohst algorithm (Cholesky).
        """
        RR = self.RR
        m = G.nrows()
        R = matrix(RR, m, m)
        for i in range(m):
            for j in range(i, m):
                s = RR(G[i, j])
                for k in range(i):
                    s -= R[k, i] * R[k, j]
                if i == j:
                    R[i, i] = sqrt(s)
                else:
                    R[i, j] = s / R[i, i]
        y = R.transpose().solve_right(vector(RR, [RR(x) for x in g]))
        c2 = RR(c) - y.dot_product(y)
        radius2 = RR(bound) - c2
        if radius2 < 0:
            return []
        results = []
        x = [ZZ(0)] * m

        def rec(i, Tprev):
            if i < 0:
                results.append(list(x))
                return
            u = RR(0)
            for j in range(i + 1, m):
                u += R[i, j] * RR(x[j])
            rem = radius2 - Tprev
            if rem < 0:
                return
            sq = sqrt(rem)
            lo = (-(u + y[i]) - sq) / R[i, i]
            hi = (-(u + y[i]) + sq) / R[i, i]
            for xi in range(int(ceil(lo)), int(floor(hi)) + 1):
                x[i] = ZZ(xi)
                val = R[i, i] * RR(xi) + u + y[i]
                rec(i - 1, Tprev + val * val)

        rec(m - 1, RR(0))
        return results

    def _enum_rank1(self, v, w, Binf):
        r"""Enumerate b = w + k*v with ||b||_inf <= Binf."""
        r = len(w)
        lo = -10 ** 30
        hi = 10 ** 30
        for i in range(r):
            if v[i] == 0:
                if abs(w[i]) > Binf:
                    return []
                continue
            if v[i] > 0:
                li = ceil((-Binf - w[i]) / v[i])
                hi_i = floor((Binf - w[i]) / v[i])
            else:
                li = ceil((Binf - w[i]) / v[i])
                hi_i = floor((-Binf - w[i]) / v[i])
            lo = max(lo, li)
            hi = min(hi, hi_i)
        return [tuple(w[i] + k * v[i] for i in range(r)) for k in range(lo, hi + 1)]

    def _enum_coset(self, basis, w, Binf, B2):
        r"""Enumerate all b in the coset w + L with ||b||_inf <= Binf."""
        Binf = ZZ(ceil(Binf))
        m = len(basis)
        if m == 0:
            if max(abs(ZZ(x)) for x in w) <= Binf:
                return [tuple(ZZ(x) for x in w)]
            return []
        if m == 1:
            return self._enum_rank1(list(basis[0]), [ZZ(x) for x in w], Binf)
        # rank >= 2: Fincke-Pohst with the 2-norm, then filter by inf-norm
        bound = ceil(self.RR(B2) ** 2)
        Bm = matrix(ZZ, [list(b) for b in basis])
        G = Bm * Bm.transpose()
        g = Bm * vector(ZZ, list(w))
        c = sum(ZZ(x) ** 2 for x in w)
        coeffs = self._enum_quadratic(G, g, c, bound)
        wv = vector(ZZ, list(w))
        result = []
        for cf in coeffs:
            b = wv + Bm.transpose() * vector(ZZ, cf)
            bt = tuple(ZZ(x) for x in b)
            if max(abs(ZZ(x)) for x in bt) <= Binf:
                result.append(bt)
        return result

    def _eval_F(self, X, Y):
        val = ZZ(0)
        for i in range(self.d + 1):
            val += self.coeffs[i] * ZZ(X) ** (self.d - i) * ZZ(Y) ** i
        return val

    def _check_candidate(self, tau, deltas, b):
        eps = prod(dj ** bi for dj, bi in zip(deltas, b))
        mu = tau * eps
        coeffs = mu.list()
        if len(coeffs) != self.d:
            return None
        if any(QQ(c) != 0 for c in coeffs[2:]):
            return None
        c0, c1 = QQ(coeffs[0]), QQ(coeffs[1])
        if c0.denominator() != 1 or c1.denominator() != 1:
            return None
        c0, c1 = ZZ(c0), ZZ(c1)
        if c0 % self.a0 != 0:
            return None
        X = c0 // self.a0
        Y = -c1
        if gcd(X, Y) != 1 or gcd(self.a0, Y) != 1:
            return None
        for XX, YY in ((X, Y), (-X, -Y)):
            Fv = self._eval_F(XX, YY)
            if Fv == 0 or Fv % self.a != 0:
                continue
            q = Fv // self.a
            if q <= 0:
                continue
            z = {}
            qq = q
            for p in self.S:
                cnt = 0
                while qq % p == 0:
                    qq //= p
                    cnt += 1
                z[p] = cnt
            if qq == 1:
                return (XX, YY, z)
        return None

    # ------------------------------------------------------------------ #
    #  Section 10.3: sieving with auxiliary rational primes
    # ------------------------------------------------------------------ #
    def _quotient_group(self, inv, hcols):
        r"""Structure of B = prod Z/inv_i Z modulo the subgroup generated by
        hcols.  Returns (qinv, qmap) where qmap maps an exponent vector of A
        to its image in B."""
        g = len(inv)
        h = len(hcols)
        M = matrix(ZZ, g, g + h)
        for i in range(g):
            M[i, i] = inv[i]
        for j in range(h):
            for i in range(g):
                M[i, g + j] = hcols[j][i]
        D, U, V = M.smith_form()
        idx = [i for i in range(g) if D[i, i] > 1]
        qinv = tuple(D[i, i] for i in idx)

        def qmap(e):
            x = U * vector(ZZ, e)
            return tuple(ZZ(x[i]) % D[i, i] for i in idx)
        return qinv, qmap

    def _cyclic_order_and_table(self, qinv, g):
        r"""Order of g in the abelian group prod Z/qinv_i, and a table
        {g*k -> k} for k in [0, order)."""
        order = ZZ(1)
        for n, gi in zip(qinv, g):
            order = lcm(order, n // gcd(n, gi))
        table = {}
        cur = tuple([ZZ(0)] * len(qinv))
        for k in range(int(order)):
            table[cur] = k
            cur = tuple((cur[i] + g[i]) % qinv[i] for i in range(len(qinv)))
        return order, table

    def _choose_aux_primes(self, num=6):
        excl = set(self.S)
        if self.a0 != 0:
            excl |= set(self.a0.prime_divisors())
        if self.a != 0:
            excl |= set(self.a.prime_divisors())
        primes = []
        p = ZZ(2)
        while len(primes) < num:
            if p not in excl:
                primes.append(p)
            p = next_prime(p)
        return primes

    def _aux_q_data(self, q, deltas, tau):
        r"""Data for the sieve modulo the auxiliary prime q (Proposition 10.3)."""
        A = self.K.ideal(q).idealstar(flag=2)
        inv = tuple(ZZ(c) for c in A.invariants())
        bid = self.K.ideal(q)._pari_bid_(2)
        nf = self.K.pari_nf()

        def dlog(e):
            return [ZZ(c) for c in nf.ideallog(e.__pari__(), bid)]

        hcols = [dlog(self.K(g)) for g in IntegerModRing(q).unit_gens()]
        qinv, qmap = self._quotient_group(inv, hcols)
        phi_deltas = [qmap(dlog(dj)) for dj in deltas]
        tl = dlog(tau)
        S = set()
        Rq = [self.a0 * u - self.theta for u in range(q)] + [self.a0]
        for r in Rq:
            try:
                d = dlog(r)
            except Exception:
                continue
            S.add(tuple(qmap([d[i] - tl[i] for i in range(len(inv))])))
        return qinv, phi_deltas, S

    def _solve_one(self, tau, deltas, S_ideals):
        r"""Fully solve one S-unit equation; returns list of (X, Y, z_dict)."""
        RR = self.RR
        r = len(deltas)
        Binf = RR(self._initial_bound(tau, deltas, S_ideals))
        B1 = RR(r) * Binf
        B2 = sqrt(RR(r)) * Binf
        k_bounds = None
        for _ in range(60):
            Bnew, B1, B2, k_bounds = self._reduce_bound_once(tau, deltas, S_ideals, Binf)
            if RR(Bnew) >= RR(Binf) * RR(1 - 1e-12):
                Binf = RR(Bnew)
                break
            Binf = RR(Bnew)
        Binf = RR(max(Binf, RR(1)))
        B2 = RR(B2)

        s = len(S_ideals)
        N = matrix(ZZ, s, r)
        for j in range(s):
            for i in range(r):
                N[j, i] = self.ord_P(deltas[i], S_ideals[j])
        kp = [self.ord_P(tau, p) for p in S_ideals]
        Lbasis = [tuple(v) for v in N.right_kernel().basis()]

        # Precompute auxiliary sieve data (only useful for rank-1 free direction)
        aux = None
        if len(Lbasis) == 1:
            v = list(Lbasis[0])
            aux = []
            for q in self._choose_aux_primes():
                try:
                    qinv, phi_deltas, S_q = self._aux_q_data(q, deltas, tau)
                except Exception:
                    continue
                g_v = tuple(sum(v[i] * phi_deltas[i][j] for i in range(r)) % qinv[j]
                            for j in range(len(qinv)))
                order, table = self._cyclic_order_and_table(qinv, g_v)
                if order > 1:
                    aux.append((qinv, phi_deltas, S_q, order, table))

        solutions = []
        for combo in product(*[range(kb + 1) for kb in k_bounds]):
            c = [combo[j] - kp[j] for j in range(s)]
            w = self._solve_linear_system(N, c)
            if w is None:
                continue
            if aux is not None and len(Lbasis) == 1:
                for k in self._sieve_k(aux, w, v, r, Binf):
                    b = tuple(w[i] + k * v[i] for i in range(r))
                    sol = self._check_candidate(tau, deltas, b)
                    if sol is not None:
                        solutions.append(sol)
            else:
                for b in self._enum_coset(Lbasis, w, Binf, B2):
                    sol = self._check_candidate(tau, deltas, b)
                    if sol is not None:
                        solutions.append(sol)
        return solutions

    def _sieve_k(self, aux, w, v, r, Binf):
        r"""Allowed values of k (free rank-1 coefficient) with |w + k v|_inf <= Binf,
        restricted by the auxiliary primes in ``aux``."""
        Binf = ZZ(ceil(Binf))
        allowed = None
        for qinv, phi_deltas, S_q, order, table in aux:
            pw = tuple(sum(w[i] * phi_deltas[i][j] for i in range(r)) % qinv[j]
                       for j in range(len(qinv)))
            residues = set()
            for t in S_q:
                d = tuple((t[j] - pw[j]) % qinv[j] for j in range(len(qinv)))
                if d in table:
                    residues.add(table[d])
            cur = set()
            for res in residues:
                lo = ceil((-Binf - res) / order)
                hi = floor((Binf - res) / order)
                for m in range(lo, hi + 1):
                    cur.add(res + m * order)
            if allowed is None:
                allowed = cur
            else:
                allowed &= cur
            if not allowed:
                return []
        if allowed is None:
            return list(range(-Binf, Binf + 1))
        return sorted(allowed)

    def solve(self):
        r"""
        Solve the Thue-Mahler equation and return the list of solutions
        (X, Y, z) where z is a dict mapping each prime in S to its exponent.
        """
        solutions = []
        seen = set()
        for tau, deltas, S_ideals in self._s_unit_equations_with_primes():
            for (X, Y, z) in self._solve_one(tau, deltas, S_ideals):
                key = (X, Y, tuple(z[p] for p in self.S))
                if key not in seen:
                    seen.add(key)
                    solutions.append((X, Y, z))
        return solutions

    def _s_unit_equations_with_primes(self):
        r"""Yield (tau, deltas, S_ideals) for every S-unit equation."""
        for a_ideal, S_ideals in self._ideal_equations():
            for tau, deltas in self._principalize(a_ideal, S_ideals):
                yield (tau, deltas, tuple(S_ideals))
