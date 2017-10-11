package main

import (
	"fmt"
	"log"
	"math"

	"github.com/gonum/matrix/mat64"

	"sort"
)

func main() {
	// construct and solve grid using Finite point method:
	const n = 10
	const nnearest = 3
	const degree = 2
	kern := GradientN{Order: 2}
	//kern := LinKernel{Slope: .3, Intercept: 7}
	bounds := Boundaries{0: GradientN{Order: 0, Rhs: 3}, (n - 1): GradientN{Order: 0, Rhs: 7}}

	pts := make([]*Point, n)
	for i := 0; i < n; i++ {
		pts[i] = NewPoint(float64(i))
	}

	basisfn := BasisFunc{Dim: len(pts[0].X), Degree: degree}

	rhs := make([]float64, len(pts))
	kp := &KernelParams{Basis: basisfn}
	for i, pref := range pts {
		kp.X = pref.X
		if bkern, ok := bounds[i]; ok {
			rhs[i] = bkern.RHS(kp)
		} else {
			rhs[i] = kern.RHS(kp)
		}
	}

	A := mat64.NewDense(len(pts), len(pts), nil)
	for i, pref := range pts {
		xref := pref.X
		kp.X = xref

		indices, nearest := Nearest(nnearest, xref, pts)

		farthest := nearest[len(nearest)-1]
		rho := 0.0
		for j := range farthest.X {
			diff := farthest.X[j] - xref[j]
			rho += diff * diff
		}

		// set weight function support distance to 1.5 times the distance to the farthest point in
		// the reference point's neighborhood.
		weightfn := NormGauss{Rho: 1.5 * math.Sqrt(rho), Epsilon: .1}

		fmt.Printf("reference point %v, x = %v\n", i+1, xref)
		fmt.Printf("    indices = %v\n", indices)

		pref.SetNeighbors(weightfn, basisfn, nearest)
		lambda := pref.LambdaMatrix()
		fmt.Printf("    lambda%v=\n% .3v\n", i+1, mat64.Formatted(lambda))

		for k, j := range indices {
			// j is the global index of the k'th local node for the approximation of the
			// neighborhood around global node/point i (i.e. xref).
			kp.Lambdas = make([]float64, basisfn.NumMonomials())
			for m := range kp.Lambdas {
				kp.Lambdas[m] = lambda.At(m, k)
			}
			fmt.Printf("        lambdas=%v\n", kp.Lambdas)
			if bkern, ok := bounds[i]; ok {
				A.Set(i, j, bkern.LHS(kp)*weightfn.Weight(xref, nearest[k].X))
			} else {
				A.Set(i, j, kern.LHS(kp)*weightfn.Weight(xref, nearest[k].X))
			}
		}
	}
	fmt.Printf("A=\n% .1v\n", mat64.Formatted(A))

	// need to add boundary conditions

	var soln mat64.Vector
	err := soln.SolveVec(A, mat64.NewVector(len(rhs), rhs))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("rhs=", rhs)
	fmt.Println("x=", soln.RawVector().Data)

	for i, val := range soln.RawVector().Data {
		pts[i].Phi = val
		pts[i].SolveCoeffs()
		fmt.Println(i, val)
	}
}

func L2DistSquared(a, b []float64) float64 {
	tot := 0.0
	for i := range a {
		diff := a[i] - b[i]
		tot += diff * diff
	}
	return tot
}

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
