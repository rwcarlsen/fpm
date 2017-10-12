package main

import (
	"flag"
	"fmt"
	"log"
)

var dbg = flag.Bool("dbg", false, "true to print debug information")

var debug = func(string, ...interface{}) (int, error) { return 0, nil }

func main() {
	flag.Parse()
	if *dbg {
		debug = fmt.Printf
	}

	// construct and solve grid using Finite point method:
	const n = 11
	const nnearest = 3
	const degree = 2
	const supportMult = 1.05
	const epsilon = 15

	//const min, max = 0, 4
	//const thermalConduc = 2
	//const heatsource = 50
	//left := Dirichlet(0)
	//right := Neumann(5 / -thermalConduc)
	//kern := KernelMult{Mult: -thermalConduc, Kernel: Poisson(heatsource)}
	//bounds := Boundaries{0: left, (n - 1): right}

	const min, max = 0, 4
	left := Dirichlet(0)
	right := Neumann(0)
	kern := Poisson(1)
	bounds := Boundaries{0: left, (n - 1): right}

	pts := make([]*Point, n)
	for i := 0; i < n; i++ {
		pts[i] = NewPoint(min + (max-min)*float64(i)/float64(n-1))
	}
	points := NewPointSet(pts)

	basisfn := BasisFunc{Dim: 1, Degree: degree}

	BuildNeighborhoods(points, nnearest, basisfn, supportMult, epsilon)
	err := Solve(points, kern, bounds)
	if err != nil {
		log.Fatal(err)
	}

	for _, p := range pts {
		for j := 0; j < 10; j++ {
			x := []float64{p.X[0] + (max-min)/10.0*float64(j)/float64(10)}
			val := p.Interpolate(x)
			fmt.Printf("%.5v\t%.5v\n", x[0], val)
		}
	}
}
