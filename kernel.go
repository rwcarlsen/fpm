package main

type Kernel interface {
	LHS(kp *KernelParams) float64
	RHS(kp *KernelParams) float64
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

// GradU represents the gradient operator acting on the dependent variable.
func (kp *KernelParams) GradU() float64 {
	tot := 0.0
	for d := 0; d < kp.Basis.Dim; d++ {
		derivs := make([]int, kp.Basis.Dim)
		derivs[d] = 1
		tot += kp.Term(derivs...)
	}
	return tot
}

// LaplaceU represents the laplace operator acting on the dependent variable.
func (kp *KernelParams) LaplaceU() float64 {
	tot := 0.0
	for d := 0; d < kp.Basis.Dim; d++ {
		derivs := make([]int, kp.Basis.Dim)
		derivs[d] = 2
		tot += kp.Term(derivs...)
	}
	return tot
}

// Term calculates the multiplier for an equation term representing a single partial derivative
// specified by derivOrders; each entry in derivOrders if the degree of (partial) derivative for
// each dimension or independent variable.
func (kp *KernelParams) Term(derivOrders ...int) float64 {
	i, mult := kp.Basis.TermAtZero(derivOrders...)
	return mult * kp.Lambdas[i]
}

type SumKernel struct{ Kernels []Kernel }

func NewSumKernel(kernels ...Kernel) SumKernel { return SumKernel{Kernels: kernels} }
func (k SumKernel) LHS(kp *KernelParams) float64 {
	tot := 0.0
	for _, kern := range k.Kernels {
		tot += kern.LHS(kp)
	}
	return tot
}
func (k SumKernel) RHS(kp *KernelParams) float64 {
	tot := 0.0
	for _, kern := range k.Kernels {
		tot += kern.RHS(kp)
	}
	return tot
}

type Poisson float64

func (p Poisson) LHS(kp *KernelParams) float64 { return kp.LaplaceU() }
func (p Poisson) RHS(kp *KernelParams) float64 { return float64(p) }

type GradientN struct {
	Order int
	Rhs   float64
}

func (g GradientN) LHS(kp *KernelParams) float64 {
	// del squared phi - need to do each dimension
	tot := 0.0
	for d := 0; d < kp.Basis.Dim; d++ {
		derivs := make([]int, kp.Basis.Dim)
		derivs[d] = g.Order
		tot += kp.Term(derivs...)
	}
	return tot
}
func (g GradientN) RHS(kp *KernelParams) float64 { return g.Rhs }

type Dirichlet float64

func (k Dirichlet) LHS(kp *KernelParams) float64 {
	if kp.Index == 0 {
		return 1
	}
	return 0
}
func (k Dirichlet) RHS(kp *KernelParams) float64 { return float64(k) }

type Neumann float64

func (k Neumann) LHS(kp *KernelParams) float64 { return kp.GradU() }
func (k Neumann) RHS(kp *KernelParams) float64 { return float64(k) }

type KernelMult struct {
	Kernel
	Mult float64
}

func (g KernelMult) LHS(kp *KernelParams) float64 { return g.Mult * g.Kernel.LHS(kp) }
