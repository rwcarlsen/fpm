package main

import (
	"fmt"
	"math"
	"testing"
)

func TestSolve_1D(t *testing.T) {
	for i, test := range SampleProblems1D {
		pointset := test.Run()
		t.Run(fmt.Sprintf("case%v", i+1), testSolve1D(pointset, test))
	}
}

func testSolve1D(got *PointSet, test SampleProblem1D) func(t *testing.T) {
	return func(t *testing.T) {
		nsamples := 11
		for i := 0; i < nsamples; i++ {
			x := []float64{test.Min + (test.Max-test.Min)*float64(i)/float64(nsamples-1)}
			v := got.Interpolate(x)
			want := test.Want(x[0])
			if math.Abs(want-v) > test.Tol {
				t.Errorf("sample %v f(x=%v): want %v, got %v", i+1, x[0], want, v)
			}
		}
	}
}
