package main

type Kernel struct {
	LHS KernelTerm
	RHS KernelTerm
}

type KernelTerm interface {
	Compute(kp *KernelParams) float64
}

type KernelList []Kernel

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
	matches := b.locIndex(kp.X)
	if len(matches) == 0 {
		return 0
	}

	return b.Vals[matches[0]]
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

type SubdomainId int

type Subdomainer interface {
	Subdomain(x []float64) []SubdomainId
}

type BoxSubdomains struct {
	Lower [][]float64
	Uper  [][]float64
	Ids   []SubdomainId
}

func (b *BoxSubdomains) Add(lower, uper []float64, id SubdomainId) {
	b.Lower = append(b.Lower, lower)
	b.Uper = append(b.Uper, uper)
	b.Ids = append(b.Ids, id)
}

func (b *BoxSubdomains) Subdomain(xs []float64) []SubdomainId {
	var matches []SubdomainId
outer:
	for i := range b.Lower {
		low := b.Lower[i]
		up := b.Uper[i]
		for j, x := range xs {
			if x < low[j] || up[j] < x {
				continue outer
			}
		}
		matches = append(matches, b.Ids[i])
	}
	return matches
}

type ZeroOutsideSubdomain struct {
	Subdomain SubdomainId
	Subdomainer
}

func (z *ZeroOutsideSubdomain) Compute(kp *KernelParams) float64 {
	a := z.Subdomainer.Subdomain(kp.X)
	b := z.Subdomainer.Subdomain(kp.StarX)
	if len(a) != len(b) || (len(a) > 0 && a[0] != b[0]) {
		return 0
	}
	return 1
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
