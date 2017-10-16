package main

type Kernel struct {
	LHS KernelTerm
	RHS KernelTerm
}

type KernelTerm interface {
	Compute(kp *KernelParams) float64
}

type Boundaries map[int]Kernel

type KernelParams struct {
	// StarIndex holds the global ID/index of the current star/primary point/node for which
	// neighborhood points are being operated on.
	StarIndex int
	// StarX holds the cordinates of the current star/primary point/node for which neighborhood
	// points are being operated on.
	StarX []float64
	// Index holds the global ID/index of the current local neighborhood point/node being operated
	// on.
	Index int
	// X holds the coordinates of the current local neighborhood point/node being operated on.
	X []float64
	// Basis holds the basis function for the current local node.  This can be used to determine
	// monomial coefficients, compute monomial values, etc.  This must be used - directly or
	// indirectly (via the KernelParams.Term method) - to calculate derivative terms for kernels.
	Basis BasisFunc
	// Lambdas holds the wieghted least squares regression coefficients for the monomial terms of
	// the current star+local point/node combination.  The nth item in the slice is the nth monomial
	// coefficient.
	Lambdas []float64
}

// TermMult calculates the multiplier for an equation term representing a single partial derivative
// specified by derivOrders; each entry in derivOrders if the degree of (partial) derivative for
// each dimension or independent variable.
func (kp *KernelParams) TermMult(derivOrders ...int) float64 {
	i, mult := kp.Basis.TermAtZero(derivOrders...)
	return mult * kp.Lambdas[i]
}

type KernelSum struct{ Terms []KernelTerm }

func NewKernelSum(terms ...KernelTerm) KernelSum { return KernelSum{Terms: terms} }
func (k KernelSum) Compute(kp *KernelParams) float64 {
	tot := 0.0
	for _, term := range k.Terms {
		tot += term.Compute(kp)
	}
	return tot
}

type KernelMult []KernelTerm

func NewKernelMult(terms ...KernelTerm) KernelMult { return KernelMult(terms) }
func (k KernelMult) Compute(kp *KernelParams) float64 {
	tot := 1.0
	for _, term := range k {
		tot *= term.Compute(kp)
	}
	return tot
}

type BoxLocation struct {
	Lower [][]float64
	Uper  [][]float64
	Vals  []float64
}

func (b *BoxLocation) Add(lower, uper []float64, val float64) {
	b.Lower = append(b.Lower, lower)
	b.Uper = append(b.Uper, uper)
	b.Vals = append(b.Vals, val)
}

func (b *BoxLocation) locIndex(loc []float64) []int {
	var matches []int
outer:
	for i := range b.Lower {
		low := b.Lower[i]
		up := b.Uper[i]
		for j, x := range loc {
			if x < low[j] || up[j] < x {
				continue outer
			}
		}
		matches = append(matches, i)
	}
	return matches
}

func (b *BoxLocation) Compute(kp *KernelParams) float64 {
	matches := b.locIndex(kp.StarX)
	if len(matches) > 1 {
		matches = b.locIndex(kp.X)
	}

	if len(matches) == 0 {
		return 0
	}

	tot := 0.0
	for _, index := range matches {
		tot += b.Vals[index]
	}
	debug("k(starx=%v,x=%v)=%v\n", kp.StarX, kp.X, tot/float64(len(matches)))
	return tot / float64(len(matches))
}

// GradU represents the gradient operator acting on the dependent variable for the current
// location specified in kp.
type GradU struct{}

func (GradU) Compute(kp *KernelParams) float64 {
	tot := 0.0
	for d := 0; d < kp.Basis.Dim; d++ {
		derivs := make([]int, kp.Basis.Dim)
		derivs[d] = 1
		tot += kp.TermMult(derivs...)
	}
	return tot
}

// LaplaceU represents the laplace operator acting on the dependent variable for the current
// location specified in kp.
type LaplaceU struct{}

func (LaplaceU) Compute(kp *KernelParams) float64 {
	tot := 0.0
	for d := 0; d < kp.Basis.Dim; d++ {
		derivs := make([]int, kp.Basis.Dim)
		derivs[d] = 2
		tot += kp.TermMult(derivs...)
	}
	return tot
}

type GradientN int

func (g GradientN) Compute(kp *KernelParams) float64 {
	// del squared phi - need to do each dimension
	tot := 0.0
	for d := 0; d < kp.Basis.Dim; d++ {
		derivs := make([]int, kp.Basis.Dim)
		derivs[d] = int(g)
		tot += kp.TermMult(derivs...)
	}
	return tot
}

type ConstKernel float64

func (k ConstKernel) Compute(kp *KernelParams) float64 { return float64(k) }

// ZeroAtNeighbors represents a kernel term that takes on its float value only at the star-point
// in a neighborhood and is zero at all other points in the neighborhood.  Useful for
// dirichlet-like constraints/boundary-conditions.
type ZeroAtNeighbors float64

func (k ZeroAtNeighbors) Compute(kp *KernelParams) float64 {
	if kp.Index == kp.StarIndex {
		return float64(k)
	}
	return 0
}

func Dirichlet(rhs float64) Kernel { return Kernel{LHS: ZeroAtNeighbors(1), RHS: ConstKernel(rhs)} }
func Neumann(rhs float64) Kernel   { return Kernel{LHS: GradU{}, RHS: ConstKernel(rhs)} }
