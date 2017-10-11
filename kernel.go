package main

type KernelParams struct {
	LocalIndex int
	X          []float64
	Basis      BasisFunc
	Lambdas    []float64
}

func (kp *KernelParams) Term(derivOrders ...int) float64 {
	i, mult := kp.Basis.TermAtZero(derivOrders...)
	return mult * kp.Lambdas[i]
}

type Boundaries map[int]Kernel

type Kernel interface {
	LHS(kp *KernelParams) float64
	RHS(kp *KernelParams) float64
}

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
	if kp.LocalIndex == 0 {
		return 1
	}
	return 0
}
func (k Dirichlet) RHS(kp *KernelParams) float64 { return float64(k) }

type KernelMult struct {
	Kernel
	Mult float64
}

func (g KernelMult) LHS(kp *KernelParams) float64 { return g.Mult * g.Kernel.LHS(kp) }
