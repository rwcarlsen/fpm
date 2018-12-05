package main

import "log"

type SampleProb1D struct {
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
	Points      []*Point
}

func (test SampleProb1D) Run() *PointSet {
	basisfn := &BasisFunc{Dim: 1, Degree: test.Degree}

	pts := test.Points
	if len(pts) == 0 {
		pts = make([]*Point, test.N)
		for i := 0; i < test.N; i++ {
			pts[i] = NewPoint(basisfn, test.Min+(test.Max-test.Min)*float64(i)/float64(test.N-1))
		}
	} else {
		for _, p := range pts {
			p.Basis = basisfn
		}
	}

	kernels := KernelList{test.Left, test.Right, test.Kernel}
	subs := &BoxSubdomains{}
	subs.Add(pts[0].X, pts[0].X, 0)
	subs.Add(pts[len(pts)-1].X, pts[len(pts)-1].X, 1)
	subs.Add(pts[0].X, pts[len(pts)-1].X, 2)

	set := NewPointSet(pts)
	set.ComputeNeighbors(&NearestN{N: test.Nnearest, Epsilon: test.Epsilon, Support: test.Support})
	err := Solve(set, subs, kernels)
	if err != nil {
		log.Fatal(err)
	}
	return set
}

var SampleProblems1D = []SampleProb1D{
	{
		Name: "Laplace",
		N:    10, Nnearest: 3, Degree: 2, Support: 1.05, Epsilon: 15,
		Min: 0, Max: 1,
		Kernel: Kernel{LHS: LaplaceU{}, RHS: ConstKernel(0)},
		Left:   Dirichlet(0),
		Right:  Dirichlet(1),
		Want:   func(x float64) float64 { return x },
		Tol:    1e-8,
	}, {
		// the discontinuous derivative due to changing multiplers across the domain necessitates
		// an interface node/point on the boundary where the discontinuity occurs.  Also, value
		// selection for the multiplier must be carefully done in order to account for a neighbor
		// node being on the boundary between both (discontinuity) regions where we need to use
		// the multiplier value for the star-node's side of the boundary.
		Name: "Laplace_Discontin",
		N:    11, Nnearest: 3, Degree: 2, Support: 1.05, Epsilon: 15,
		Min: 0, Max: 1,
		Kernel: Kernel{
			LHS: NewKernelMult(LaplaceU{}, &BoxLocation{
				Lower: [][]float64{{0}, {.5}},
				Uper:  [][]float64{{.5}, {1}},
				Vals:  []float64{1, 2}},
			),
			RHS: ConstKernel(0),
		},
		Left:  Dirichlet(0),
		Right: Dirichlet(1),
		Want: func(x float64) float64 {
			if x < .5 {
				return 4.0 / 3 * x
			}
			return 4.0/3*.5 + 2.0/3*(x-.5)
		},
		Tol: 1e-8,
		//Points: []*Point{
		//	NewPoint(nil, 0),
		//	NewPoint(nil, .1),
		//	NewPoint(nil, .2),
		//	NewPoint(nil, .3),
		//	NewPoint(nil, .4),
		//	NewPoint(nil, .5),
		//	NewPoint(nil, .6),
		//	NewPoint(nil, .7),
		//	NewPoint(nil, .8),
		//	NewPoint(nil, .9),
		//	NewPoint(nil, 1),
		//},
	}, {
		Name: "Poisson",
		N:    10, Nnearest: 3, Degree: 2, Support: 1.05, Epsilon: 15,
		Min: 0, Max: 1,
		Kernel: Kernel{LHS: LaplaceU{}, RHS: ConstKernel(10)},
		Left:   Dirichlet(0),
		Right:  Dirichlet(0),
		Want:   func(x float64) float64 { return 5 * x * (x - 1) },
		Tol:    1e-8,
	}, {
		Name: "Poisson_Neumann",
		N:    10, Nnearest: 3, Degree: 2, Support: 1.05, Epsilon: 15,
		Min: 0, Max: 1,
		Kernel: Kernel{LHS: LaplaceU{}, RHS: ConstKernel(10)},
		Left:   Dirichlet(0),
		Right:  Neumann(1),
		Want:   func(x float64) float64 { return 5*x*x - 9*x },
		Tol:    1e-8,
	},
}
