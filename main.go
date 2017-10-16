package main

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

	if *prob != "" {
		RunSample(*prob)
	}
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

func printSolution(w io.Writer, set *PointSet, prob SampleProblem1D) {
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
