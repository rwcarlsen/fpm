package main

import (
	"math"
	"sort"

	"github.com/gonum/matrix/mat64"
)

func Nearest(n int, x []float64, pts []*Point) (indices []int, nearest []*Point) {
	sorted := make([]int, len(pts))
	for i := range sorted {
		sorted[i] = i
	}
	sort.Slice(sorted, func(i, j int) bool {
		return L2DistSquared(pts[sorted[i]].X, x) < L2DistSquared(pts[sorted[j]].X, x)
	})

	nearest = make([]*Point, n)
	indices = make([]int, n)
	for i := range nearest {
		indices[i] = sorted[i]
		nearest[i] = pts[sorted[i]]
	}
	return indices, nearest
}

func L2DistSquared(a, b []float64) float64 {
	tot := 0.0
	for i := range a {
		diff := a[i] - b[i]
		tot += diff * diff
	}
	return tot
}

func BuildNeighborhoods(set *PointSet, nnearest int, bf BasisFunc, support, epsilon float64) {
	kp := &KernelParams{}
	for _, pref := range set.Points() {
		kp.X = pref.X
		_, nearest := Nearest(nnearest, pref.X, set.Points())

		farthest := nearest[len(nearest)-1]
		dist := 0.0
		for j := range farthest.X {
			diff := farthest.X[j] - pref.X[j]
			dist += diff * diff
		}
		dist = math.Sqrt(dist)

		weightfn := NormGauss{Rho: support * dist, Epsilon: epsilon}
		pref.SetNeighbors(weightfn, bf, nearest)
	}
}

func Solve(set *PointSet, kernel Kernel, bounds Boundaries) error {
	points := set.Points()
	kp := &KernelParams{}

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

	// need to add boundary conditions

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
