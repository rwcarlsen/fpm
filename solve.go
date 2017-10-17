package main

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

func Solve(set *PointSet, kernel Kernel, bounds Boundaries) error {
	points := set.Points()
	kp := &KernelParams{}

	// build right hand side (i.e. does not depend on dependent var)
	rhs := make([]float64, set.Len())
	for i, pref := range points {
		kp.StarIndex = i
		kp.StarX = pref.X
		kp.X = pref.X
		kp.Index = i
		kp.Basis = pref.Basis

		kern := kernel
		if bkern, ok := bounds[i]; ok {
			kern = bkern
		}
		rhs[i] = kern.RHS.Compute(kp)
	}

	// build left hand side (does depend on dependent var)
	A := mat64.NewDense(set.Len(), set.Len(), nil)
	for i, pref := range points {
		kp.Basis = pref.Basis
		kp.StarIndex = i
		kp.StarX = pref.X

		lambda := pref.LambdaMatrix()
		debug("xref=%v, lambda=\n% .2v\n", pref.X, mat64.Formatted(lambda))

		kern := kernel
		if bkern, ok := bounds[i]; ok {
			kern = bkern
		}

		for k, neighbor := range pref.Neighbors {
			// j is the global index of the k'th local node for the approximation of the
			// neighborhood around global node/point i (i.e. x-reference).
			j := neighbor.Index
			kp.Index = j
			kp.X = neighbor.X

			if len(kp.Lambdas) != kp.Basis.NumMonomials() {
				kp.Lambdas = make([]float64, kp.Basis.NumMonomials())
			}
			for m := range kp.Lambdas {
				kp.Lambdas[m] = lambda.At(m, k)
			}
			A.Set(i, j, kern.LHS.Compute(kp)*math.Sqrt(pref.W.Weight(pref.X, neighbor.X)))
		}
	}
	debug("A=\n% .3v\n", mat64.Formatted(A))
	debug("rhs=%.3v\n", rhs)

	// solve, pass solved values to points, and compute monomial coefficients
	var soln mat64.Vector
	err := soln.SolveVec(A, mat64.NewVector(len(rhs), rhs))
	if err != nil {
		return err
	}
	debug("phi=%.4v\n", soln.RawVector().Data)

	for i, val := range soln.RawVector().Data {
		points[i].Phi = val
	}
	for _, p := range points {
		p.SolveCoeffs()
	}
	return nil
}
