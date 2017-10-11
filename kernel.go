package main

type KernelParams struct {
	Index   int
	X       []float64
	Basis   BasisFunc
	Lambdas []float64
}

type Boundaries map[int]Kernel

type Kernel interface {
	LHS(kp *KernelParams) float64
	RHS(kp *KernelParams) float64
}

type LinKernel struct {
	Slope     float64
	Intercept float64
}

func (k LinKernel) LHS(kp *KernelParams) float64 { return GradientN{Order: 0}.LHS(kp) }
func (k LinKernel) RHS(kp *KernelParams) float64 { return k.Slope*kp.X[0] + k.Intercept }

type GradientN struct {
	Order int
	Rhs   float64
}

type KernelMult struct {
	Kernel
	Mult float64
}

func (g KernelMult) LHS(kp *KernelParams) float64 { return g.Mult * g.Kernel.LHS(kp) }

func (g GradientN) LHS(kp *KernelParams) float64 {
	// del squared phi - need to do each dimension
	tot := 0.0
	for d := 0; d < kp.Basis.Dim; d++ {
		derivs := make([]int, kp.Basis.Dim)
		derivs[d] = g.Order
		i, mult := kp.Basis.TermAtZero(derivs...)
		tot += mult * kp.Lambdas[i]
	}
	return tot
}
func (g GradientN) RHS(kp *KernelParams) float64 { return g.Rhs }
