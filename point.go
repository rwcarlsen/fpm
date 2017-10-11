package main

import (
	"fmt"
	"log"
	"math"

	"github.com/gonum/matrix/mat64"
)

type Point struct {
	X []float64
	// weights is the weight for each neighbor of X
	weights   []float64
	neighbors []*Point
	Phi       float64
	coeffs    []float64
	bf        BasisFunc
	wf        WeightFunc
}

func NewPoint(x ...float64) *Point { return &Point{X: x} }

// SolveCoeffs computes the approximated solution's monomial coefficients using the Phi values of
// all nodes in this point's neighborhood (i.e. Lambda*[w_k*phi_k]).  All Phi values for this node
// and its neighbors must have already been calculated+set.
func (p *Point) SolveCoeffs() {
	v := mat64.NewVector(len(p.neighbors), nil)
	for i, neighbor := range p.neighbors {
		v.SetVec(i, neighbor.Phi*p.weights[i])
	}

	p.coeffs = make([]float64, p.bf.NumMonomials())
	soln := mat64.NewVector(len(p.coeffs), p.coeffs)
	soln.MulVec(p.LambdaMatrix(), v)
	fmt.Println("coeffs of ", p.X, ":", p.coeffs)
}

func (p *Point) Interpolate(x []float64) float64 {
	xrel := make([]float64, len(p.X))
	for i := range x {
		xrel[i] = x[i] - p.X[i]
	}

	tot := 0.0
	for i, coeff := range p.coeffs {
		tot += p.bf.MonomialVal(i, xrel) * coeff
	}
	return tot
}

func (p *Point) LambdaMatrix() *mat64.Dense {
	r, c := len(p.neighbors), p.bf.NumMonomials()
	A := mat64.NewDense(r, c, nil)

	for k, neighbor := range p.neighbors {
		xrel := make([]float64, len(p.X))
		for i := range neighbor.X {
			xrel[i] = neighbor.X[i] - p.X[i]
		}

		fmt.Printf("    xrel=%v, weight=%v\n", xrel, p.weights[k])
		for i := 0; i < c; i++ {
			A.Set(k, i, p.weights[k]*p.bf.MonomialVal(i, xrel))
			fmt.Printf("        LambdaMat[%v,%v]=%v, monomial=%v\n", k, i, A.At(k, i), p.bf.MonomialVal(i, xrel))
		}
	}

	var tmp mat64.Dense
	tmp.Mul(A.T(), A)
	fmt.Printf("    A=\n% .3v\n", mat64.Formatted(A))
	fmt.Printf("    A^T*A=\n% .3v\n", mat64.Formatted(&tmp))

	var lambda mat64.Dense
	err := lambda.Solve(&tmp, A.T())
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("    lambda=\n% .3v\n", mat64.Formatted(&lambda))
	return &lambda
}

// SetNeighbors tells the point what other points are in its local neighborhood and initiates the
// computation of the Lambda matrix used for interpolating and building the global system to solve
// the differential equation(s).
func (p *Point) SetNeighbors(wf WeightFunc, bf BasisFunc, neighbors []*Point) {
	p.neighbors = neighbors
	p.bf, p.wf = bf, wf

	p.weights = make([]float64, len(neighbors))
	for k, neighbor := range neighbors {
		p.weights[k] = math.Sqrt(wf.Weight(p.X, neighbor.X))
	}
}
