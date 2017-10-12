package main

import "log"

type SampleProblem1D struct {
	Name        string
	N           int
	Nnearest    int
	Degree      int
	Support     float64
	Epsilon     float64
	Min, Max    float64
	Tol         float64
	Kernel      Kernel
	Left, Right Kernel
	Want        func(x float64) float64
}

func (test SampleProblem1D) Run() *PointSet {
	bounds := Boundaries{0: test.Left, (test.N - 1): test.Right}

	pts := make([]*Point, test.N)
	for i := 0; i < test.N; i++ {
		pts[i] = NewPoint(test.Min + (test.Max-test.Min)*float64(i)/float64(test.N-1))
	}
	points := NewPointSet(pts)

	basisfn := BasisFunc{Dim: 1, Degree: test.Degree}

	BuildNeighborhoods(points, test.Nnearest, basisfn, test.Support, test.Epsilon)
	err := Solve(points, test.Kernel, bounds)
	if err != nil {
		log.Fatal(err)
	}
	return points
}

var SampleProblems1D = []SampleProblem1D{
	{
		Name: "Laplace",
		N:    10, Nnearest: 3, Degree: 2, Support: 1.05, Epsilon: 15,
		Min: 0, Max: 1,
		Kernel: Poisson(0),
		Left:   Dirichlet(0),
		Right:  Dirichlet(1),
		Want:   func(x float64) float64 { return x },
		Tol:    1e-8,
	}, {
		Name: "Poisson",
		N:    10, Nnearest: 3, Degree: 2, Support: 1.05, Epsilon: 15,
		Min: 0, Max: 1,
		Kernel: Poisson(10),
		Left:   Dirichlet(0),
		Right:  Dirichlet(0),
		Want:   func(x float64) float64 { return 5 * x * (x - 1) },
		Tol:    1e-8,
	}, {
		Name: "Poisson_Neumann",
		N:    10, Nnearest: 3, Degree: 2, Support: 1.05, Epsilon: 15,
		Min: 0, Max: 1,
		Kernel: Poisson(10),
		Left:   Dirichlet(0),
		Right:  Neumann(1),
		Want:   func(x float64) float64 { return 5*x*x - 9*x },
		Tol:    1e-8,
	},
}
