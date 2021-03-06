package main

// TODO:
//    * figure out how to deal with discontinuities in the differential equation.  It seems like
//      there needs to be a node *on* the interface/discontinuity.
//    * figure out how to avoid singular global matrices due to bad neighbor groupings.  Danger
//      zones include when two or more nodes/points have identical neighbor sets.
//    *

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"log"
	"os/exec"
	"strings"
)

var dbg = flag.Bool("dbg", false, "true to print debug information")
var plot = flag.String("plot", "", "generate and write a plot with gnuplot to a file")
var nsoln = flag.Int("nsol", 10, "number of uniformly distributed points to sample+print solution over")
var prob = flag.String("prob", "", "name of test problem to run")

var debug = func(string, ...interface{}) (int, error) { return 0, nil }

func main() {
	flag.Parse()
	if *dbg {
		debug = fmt.Printf
	}

	//wfn := NormGauss{Rho: 1, Epsilon: 15}
	//n := 10
	//for i := 0; i <= n; i++ {
	//	x := float64(i) / float64(n)
	//	w := wfn.Weight([]float64{0}, []float64{x})
	//	fmt.Printf("weight[x=%v]=%v\n", x, w)
	//}
	//return

	if *prob != "" {
		RunSample(*prob)
	}
}

//func InterfaceProblem() {
//	const n = 11
//	const min, max = 0, 1
//	bounds := Boundaries{0: Dirichlet(0), (n - 1): Dirichlet(1)}
//	basisfn := BasisFunc{Dim: 1, Degree: 2}
//	kernel := Kernel{
//		LHS: NewKernelMult(LaplaceU{}, &BoxLocation{
//			Lower: [][]float64{{0}, {.5}},
//			Uper:  [][]float64{{.5}, {1}},
//			Vals:  []float64{1, 2}},
//		),
//		RHS: ConstKernel(0),
//	}
//
//	debug("points\n")
//	pts := make([]*Point, n)
//	for i := 0; i < n; i++ {
//		pts[i] = NewPoint(basisfn, min+(max-min)*float64(i)/float64(n-1))
//		debug("    %.3v\n", pts[i].X)
//	}
//	points := NewPointSet(pts)
//
//	points.ComputeNeighbors(&NearestN{N: 4, Epsilon: 15, Support: 1.05})
//	err := points.Solve(kernel, bounds)
//	if err != nil {
//		log.Fatal(err)
//	}
//}

func RunSample(name string) {
	for _, prob := range SampleProblems1D {
		if strings.ToLower(prob.Name) == strings.ToLower(name) {
			pointset := prob.Run()
			var buf bytes.Buffer
			printSolution(&buf, pointset, prob)
			if *plot == "" {
				log.Print("Solution:")
				fmt.Print(buf.String())
			} else {
				fmt.Fprintf(&buf, "e\n")
				printSolution(&buf, pointset, prob)
				cmd := exec.Command("gnuplot", "-e", `set terminal svg; set output "`+*plot+`"; plot "-" u 1:2 w l title "FPM Approximation", "-" u 1:3 w l title "Analytical Solution`)
				cmd.Stdin = &buf
				err := cmd.Run()
				if err != nil {
					log.Fatal(err)
				}
			}
			return
		}
	}
	log.Fatalf("sample problem %v not found", name)
}

func printSolution(w io.Writer, set *PointSet, prob SampleProb1D) {
	n := 1
	dims := make([]int, n)
	for i := range dims {
		dims[i] = *nsoln + 1
	}

	perms := Permute(0, dims...)
	for _, p := range perms {
		x := make([]float64, len(p))
		for i, ii := range p {
			x[i] = float64(ii)/float64(*nsoln)*(prob.Max-prob.Min) + prob.Min
		}

		u := set.Interpolate(x)
		fmt.Fprintf(w, "%.3v\t%.3v\t%.3v\n", x[0], u, prob.Want(x[0]))
	}
}
