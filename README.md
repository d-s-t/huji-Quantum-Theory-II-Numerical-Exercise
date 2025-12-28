# Numerical Exercise
**Course:** Quantum Mechanics II, The Hebrew University
**Author:** David Shem-Tov

This project contains the Numerov method exercise. We will numerically calculate the bond energy of a system consisting of several electrons using the Numerov method, working in **Planck natural units**.

---

## Background
A particle in a radial potential $V(r)$ satisfies the Schrödinger equation:

$$
\left[ -\frac{1}{2} \nabla^2 + V(r) \right]\psi(r) = E\psi(r)
$$

Substituting $\psi(r) = \frac{u(r)}{r} Y_{l,m}(\theta, \phi)$, we obtain the radial equation:

$$
-\frac{1}{2} \frac{d^2 u(r)}{dr^2} + \left[ V(r) + \frac{l(l+1)}{2r^2} \right]u(r) = Eu(r)
$$

**Boundary Conditions:**
* $u(0) = 0$ (where $u(r \to 0) \sim r^{l+1}$)
* $u(r \to \infty) = 0$

*Note: Even though $u$ is dependent on $n$ and $l$, we will omit these indices for clarity ($u_{nl}(r) \to u(r)$).*

The radial equation can be rewritten as:

$$
-\frac{u''(r)}{2} + W(r)u(r) = Eu(r)
$$

Where the effective potential $W(r)$ is:

$$
W(r) = V(r) + \frac{l(l+1)}{2r^2}
$$

---

## Finite Difference Method

### 1. Discretization
Define a maximum radius $R$ and divide the domain $(0, R)$ into $K$ segments. Each segment has a width of $\Delta = R/K$. This creates a grid of points $r_k = k \cdot \Delta$ and corresponding values $u_k = u(r_k)$.

### 2. Approximation (Central Difference)
Using a Taylor series expansion, the second derivative is approximated with an error order of $O(\Delta^2)$:

$$
u''_k \approx \frac{u_{k+1} - 2u_k + u_{k-1}}{\Delta^2}
$$

### 3. Eigenvalue Problem
Substituting this approximation into the Schrödinger equation yields the discrete equation for each point $k$:

$$
-\frac{1}{2\Delta^2} (u_{k+1} - 2u_k + u_{k-1}) + W_k u_k = \epsilon u_k
$$

This results in a matrix eigenvalue equation where the Hamiltonian is represented as a tridiagonal matrix (combining the kinetic energy difference terms and the potential energy $W$).

---

## Numerov Method

### 1. Higher Accuracy Goal
Unlike the standard Finite Difference method (which is 2nd-order accurate), Numerov's method aims for **4th-order accuracy** (error terms of $O(\Delta^6)$ instead of $O(\Delta^4)$).

This is achieved by combining the Taylor series expansions of the function $u$ and its second derivative $u''$ in a specific way to cancel out lower-order error terms.

### 2. The Numerov Relation
The core formula derived is:

$$
\frac{u_{k+1} + u_{k-1} - 2u_k}{\Delta^2} = \frac{1}{12} (u''_{k+1} + 10u''_k + u''_{k-1})
$$

### 3. Generalized Eigenvalue Problem
Substituting the Schrödinger equation into the Numerov relation results in a slightly more complex matrix equation:

$$
H \mathbf{u} = \epsilon N \mathbf{u}
$$

Where:
* $\mathbf{H}$ is a tridiagonal matrix containing the kinetic terms and potential energy ($W$).
* $\mathbf{N}$ is a constant tridiagonal matrix (with values derived from the $1/12, 10/12, 1/12$ coefficients).
* $\epsilon$ is the energy eigenvalue.

### 4. Implementation Note
This is a **Generalized Eigenvalue Problem**.
* It can be solved using standard linear algebra libraries capable of solving $Av = \lambda Bv$.
* Alternatively, it can be rewritten as a standard eigenvalue problem by inverting $N$:
    $$(N^{-1} H) \mathbf{u} = \epsilon \mathbf{u}$$