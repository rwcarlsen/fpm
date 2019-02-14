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
	"math"
	"os"
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

	if *prob != "" {
		RunSample(*prob)
	} else {
		InterfaceProblem()
	}
}

func InterfaceProblem() {
	nt := 10
	nx := 10
	xmin := 0.0
	xmax := 1.0
	tmin := 0.0
	tmax := 1.0
	nneighbors := 7
	degree := 2
	support := 1.05
	epsilon := 15.0

	mins := []float64{tmin, xmin}
	maxes := []float64{tmax, xmax}
	dims := []int{nt, nx}
	perms := Permute(0, dims...)

	basisfn := &BasisFunc{Dim: 2, Degree: degree}
	kernel := Kernel{
		LHS: NewKernelSum(LaplaceUSpace{}, NewKernelMult(ConstKernel(-1), GradientNTime(1))),
		RHS: ConstKernel(0),
	}

	debug("points:\n")
	pts := make([]*Point, len(perms))
	for i, p := range perms {
		x := make([]float64, len(p))
		for i, ii := range p {
			x[i] = float64(ii)/float64(dims[i]-1)*(maxes[i]-mins[i]) + mins[i]
		}
		pts[i] = NewPoint(basisfn, x...)
		debug("    %.3v\n", pts[i].X)
	}
	points := NewPointSet(pts)

	bounds := Boundaries{}
	for i, p := range pts {
		if p.X[0] == tmin {
			bounds[i] = Dirichlet(math.Sin(math.Pi*p.X[0]) + p.X[1])
		} else if p.X[1] == xmin {
			bounds[i] = Dirichlet(xmin)
		} else if p.X[1] == xmax {
			bounds[i] = Dirichlet(xmax)
		}
	}

	points.ComputeNeighbors(&NearestN{N: nneighbors, Epsilon: epsilon, Support: support})

	err := Solve(points, kernel, bounds)
	if err != nil {
		log.Fatal(err)
	}
	printSolutionND(os.Stdout, points, mins, maxes)
}

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

func printSolutionND(w io.Writer, set *PointSet, mins, maxes []float64) {
	n := len(mins)
	dims := make([]int, n)
	for i := range dims {
		dims[i] = *nsoln + 1
	}

	perms := Permute(0, dims...)
	for _, p := range perms {
		x := make([]float64, len(p))
		for i, ii := range p {
			x[i] = float64(ii)/float64(*nsoln)*(maxes[i]-mins[i]) + mins[i]
		}

		u := set.Interpolate(x)
		for _, xx := range x {
			fmt.Fprintf(w, "%.3v\t", xx)
		}
		fmt.Fprintf(w, "%.3v\n", u)
	}
}
