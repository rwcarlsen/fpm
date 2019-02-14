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
	Basis *BasisFunc
	// Lambdas holds the wieghted least squares regression coefficients for the monomial terms of
	// the current star+local point/node combination.  The nth item in the slice is the nth monomial
	// coefficient.
	Lambdas []float64
}

// TermMult calculates the multiplier for an equation term representing a single partial derivative
// specified by derivOrders; each entry in derivOrders is the degree of (partial) derivative for
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

// TODO: In order to handle discontinuities between regions of the problem domain, we need to be
// able to allow users to specify different regions of the problem  in a way that allows defining
// (star) point neighborhoods to be limited to points that are within the star point's region.
// Also we need to be able to specify special kernels that are used for star points that are
// located on the boundary between regions.  These boundary star points should basically have no
// points in their neighborhood and also need to be in the neighborhoods of star points on all
// bordering regions.  This should resolve solution artifacts and conditioning issues related to
// discontinuities and also fix issues caused mucking around with star-node support range in order
// to try to fix discontinuity weirdness.
//
// I think this should solve the problem if the star node is on an interface between
// subdomains - to compute the linear system matrix entries for it and its neighbors:
//     - If the star node is on an interface between subdomains:
//         - for neighbors, use the value of kernels in the region of the neighbor's coordinates.
//         - for the star node, use the average value of the kernel contribution from all bordering subdomains
//     - If the star node is not on an interface:
//         - for the star node, use the normal/only kernel value
//         - for neighbor nodes:
//             - on an interface use the kernel value for the star node's subdomain
//             - otherwise just use the node's normal/only kernel value

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

type FuncKernel func(kp *KernelParams) float64

func (f FuncKernel) Compute(kp *KernelParams) float64 { return f(kp) }

func (b *BoxLocation) Compute(kp *KernelParams) float64 {
	matches := b.locIndex(kp.X)
	if len(matches) == 0 {
		return 0
	} else if len(matches) > 1 {
		// if the neighbor point is on a boundary between two values, prefer the value at the star point
		matches = b.locIndex(kp.StarX)
	}

	// if there are still multiple matches (i.e. even at the star point), use the average
	sum := 0.0
	for _, m := range matches {
		sum += b.Vals[m]
	}

	return sum / float64(len(matches))
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

// LaplaceSpaceTime represents the laplace operator acting on the dependent variable for the current
// location specified in kp.
type LaplaceUSpace struct{}

func (LaplaceUSpace) Compute(kp *KernelParams) float64 {
	tot := 0.0
	for d := 1; d < kp.Basis.Dim; d++ {
		derivs := make([]int, kp.Basis.Dim)
		derivs[d] = 2
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
	tot := 0.0
	for d := 0; d < kp.Basis.Dim; d++ {
		derivs := make([]int, kp.Basis.Dim)
		derivs[d] = int(g)
		tot += kp.TermMult(derivs...)
	}
	return tot
}

type GradientNSpace int

func (g GradientNSpace) Compute(kp *KernelParams) float64 {
	tot := 0.0
	for d := 1; d < kp.Basis.Dim; d++ {
		derivs := make([]int, kp.Basis.Dim)
		derivs[d] = int(g)
		tot += kp.TermMult(derivs...)
	}
	return tot
}

type GradientNTime int

func (g GradientNTime) Compute(kp *KernelParams) float64 {
	derivs := make([]int, kp.Basis.Dim)
	derivs[0] = int(g)
	return kp.TermMult(derivs...)
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
