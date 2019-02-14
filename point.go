package main

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

type PointSet struct {
	points []*Point
}

func NewPointSet(points []*Point) *PointSet {
	for i := range points {
		points[i].Index = i
	}
	return &PointSet{points: points}
}

func (ps *PointSet) ComputeNeighbors(nh Neighborhooder) {
	for _, p := range ps.points {
		p.SetNeighbors(nh.Neighborhood(p, ps))
	}
}

func (ps *PointSet) Append(points ...*Point) {
	index := ps.points[len(ps.points)-1].Index + 1
	for i := range points {
		points[i].Index = index
		index++
	}
	ps.points = append(ps.points, points...)
}

func (ps *PointSet) Points() []*Point { return ps.points }
func (ps *PointSet) Len() int         { return len(ps.points) }

func (ps *PointSet) Interpolate(x []float64) float64 {
	n := 1
	_, nearest := Nearest(n, x, ps.points)
	sum := 0.0
	for _, p := range nearest {
		sum += p.Interpolate(x)
	}
	return sum / float64(n)
}

type Neighborhooder interface {
	Neighborhood(p *Point, set *PointSet) ([]*Point, WeightFunc)
}

type NearestN struct {
	N       int
	Epsilon float64
	Support float64
}

func (g *NearestN) Neighborhood(p *Point, set *PointSet) ([]*Point, WeightFunc) {
	_, nearest := Nearest(g.N, p.X, set.Points())
	farthest := nearest[len(nearest)-1].X
	dist := math.Sqrt(L2DistSquared(p.X, farthest))
	return nearest, NormGauss{Epsilon: g.Epsilon, Rho: g.Support * dist}
}

type Point struct {
	// The index of this point in the global (ordered) point array.
	Index     int
	X         []float64
	Neighbors []*Point
	Phi       float64
	Basis     *BasisFunc
	W         WeightFunc
	// weights is the weight for each neighbor of X - len(weights) == len(Neighbors)
	weights []float64
	coeffs  []float64
}

func NewPoint(bf *BasisFunc, x ...float64) *Point { return &Point{X: x, Index: -1, Basis: bf} }

// SolveCoeffs computes the approximated solution's monomial coefficients using the Phi values of
// all nodes in this point's neighborhood (i.e. Lambda*[w_k*phi_k]).  All Phi values for this node
// and its neighbors must have already been calculated+set.
func (p *Point) SolveCoeffs() {
	v := mat64.NewVector(len(p.Neighbors), nil)
	for i, neighbor := range p.Neighbors {
		v.SetVec(i, neighbor.Phi*p.weights[i])
	}

	p.coeffs = make([]float64, p.Basis.NumMonomials())
	soln := mat64.NewVector(len(p.coeffs), p.coeffs)
	soln.MulVec(p.LambdaMatrix(), v)
}

func (p *Point) Interpolate(x []float64) float64 {
	xrel := make([]float64, len(p.X))
	for i := range x {
		xrel[i] = x[i] - p.X[i]
	}

	tot := 0.0
	for i, coeff := range p.coeffs {
		v := p.Basis.MonomialVal(i, xrel) * coeff
		debug("monomial %.3v*x^%v at x=%v is %.3v\n", coeff, p.Basis.perms[i][0], xrel[0], v)
		tot += v
	}
	return tot
}

// LambdaMatrix represents the "(X^T*X)^-1 * X^T" matrix in the linear least squares approximation
// that fits the points in this point's neighborhood.
func (p *Point) LambdaMatrix() *mat64.Dense {
	if p.Neighbors == nil {
		panic("cannot calculate lambda matrix before neighbors are set")
	}

	r, c := len(p.Neighbors), p.Basis.NumMonomials()
	A := mat64.NewDense(r, c, nil)

	for k, neighbor := range p.Neighbors {
		xrel := make([]float64, len(p.X))
		for i := range neighbor.X {
			xrel[i] = neighbor.X[i] - p.X[i]
		}

		for i := 0; i < c; i++ {
			A.Set(k, i, p.weights[k]*p.Basis.MonomialVal(i, xrel))
		}
	}

	var tmp mat64.Dense
	tmp.Mul(A.T(), A)

	var lambda mat64.Dense
	err := lambda.Solve(&tmp, A.T())
	if err != nil {
		panic(err)
	}
	return &lambda
}

// SetNeighbors tells the point what other points are in its local neighborhood and initiates the
// computation of the Lambda matrix used for interpolating and building the global system to solve
// the differential equation(s).
func (p *Point) SetNeighbors(neighbors []*Point, wf WeightFunc) {
	p.Neighbors = neighbors
	p.W = wf

	p.weights = make([]float64, len(neighbors))
	for k, neighbor := range neighbors {
		p.weights[k] = math.Sqrt(wf.Weight(p.X, neighbor.X))
	}
}
