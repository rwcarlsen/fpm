package main

import (
	"fmt"
	"log"
	"math"

	"github.com/gonum/matrix/mat64"

	"sort"
)

type KernelParams struct {
	Index   int
	X       []float64
	Basis   BasisFunc
	Lambdas []float64
}

type Boundaries map[int]Kernel

type Kernel interface {
	LHS(kp *KernelParams) float64
	RHS(kp *KernelParams) float64
}

type GradientN struct {
	Order int
	Rhs   float64
}

type KernelMult struct {
	Kernel
	Mult float64
}

func (g KernelMult) LHS(kp *KernelParams) float64 { return g.Mult * g.Kernel.LHS(kp) }

func (g GradientN) LHS(kp *KernelParams) float64 {
	// del squared phi - need to do each dimension
	tot := 0.0
	for d := 0; d < kp.Basis.Dim; d++ {
		derivs := make([]int, kp.Basis.Dim)
		derivs[d] = g.Order
		i, mult := kp.Basis.TermAtZero(derivs...)
		tot += mult * kp.Lambdas[i]
	}
	return tot
}
func (g GradientN) RHS(kp *KernelParams) float64 { return g.Rhs }

func main() {
	// construct and solve grid using Finite point method:
	const n = 10
	const nnearest = 3
	const degree = 2
	kern := GradientN{Order: 2}
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

type WeightFunc interface {
	Weight(xref, x []float64) float64
}

// NormGauss implements a compatly supported radial basis function using the normalized Gaussian
// function.  The value of the type is the support range/distance for the neighborhood for which
// weights are being calculated.
type NormGauss struct {
	// The value of the type is the support range/distance for the neighborhood for which
	// weights are being calculated.
	Rho float64
	// Epsilon is the shape parameter
	Epsilon float64
}

func (n NormGauss) Weight(xref, x []float64) float64 {
	tot := 0.0
	for i := range x {
		diff := x[i] - xref[i]
		tot += diff * diff
	}
	dist := math.Sqrt(tot)

	if dist >= n.Rho {
		return 0
	}
	return (math.Exp(-n.Epsilon*math.Pow(dist/n.Rho, 2)) - math.Exp(-n.Epsilon)) / (1 - math.Exp(-n.Epsilon))
}

type BasisFunc struct {
	Dim int
	// Degree should generally be at least as much as the order of the differential equation being
	// solved.  More accurately, the number of monomials in the basis function needs to be greater
	// than the order of the differential equation - the number of monomials is a function of both
	// Degree and Dim.
	Degree int
	perms  [][]int
}

func (b *BasisFunc) init() {
	if b.perms == nil {
		dims := make([]int, b.Dim)
		for i := range dims {
			dims[i] = b.Degree + 1
		}
		b.perms = Permute(b.Degree, dims...)
		fmt.Println("basisfunc-monomials=", b.perms)
	}
}

// TermAtZero calculates and returns the multiplier (due to derivaties) on the monomial that
// matches the set of derivative orders (nth derivative for each dimension/variable) and its
// associated index.  This is equivalent to the partial derivative described by derivOrders of the
// basis function evaluated at X=0 (all dimensions zero).
func (b *BasisFunc) TermAtZero(derivOrders ...int) (index int, multiplier float64) {
	b.init()
	if len(derivOrders) != b.Dim {
		panic(fmt.Sprintf("wrong number of derivative orders: want %v, got %v", b.Dim, len(derivOrders)))
	}

outer:
	for i, perm := range b.perms {
		for dim, exp := range perm {
			if exp != derivOrders[dim] {
				continue outer
			}
		}
		mult := 1.0
		for _, exp := range perm {
			mult *= float64(factorial(0, exp))
		}
		return i, mult
	}
	return 0, 0
}

func (b *BasisFunc) NumMonomials() int {
	b.init()
	return len(b.perms)
}

func (b *BasisFunc) MonomialVal(i int, x []float64) float64 {
	b.init()
	if len(x) != b.Dim {
		panic(fmt.Sprintf("wrong dimension for x: want %v, got %v", b.Dim, len(x)))
	} else if i < 0 || i >= len(b.perms) {
		panic(fmt.Sprintf("invalid monomial index %v", i))
	}

	mon := b.perms[i]
	cum := 1.0
	for dim, exp := range mon {
		cum *= math.Pow(x[dim], float64(exp))
	}
	return cum
}

func (b *BasisFunc) Val(coeffs, x []float64) float64 {
	b.init()
	if len(coeffs) != len(b.perms) {
		panic(fmt.Sprintf("wrong number of coefficients: want %v, got %v", len(b.perms), len(coeffs)))
	} else if len(x) != b.Dim {
		panic(fmt.Sprintf("wrong dimension for x: want %v, got %v", b.Dim, len(x)))
	}

	tot := 0.0
	for i, perm := range b.perms {
		cum := coeffs[i]
		for dim, exp := range perm {
			cum *= math.Pow(x[dim], float64(exp))
		}
		tot += cum
	}
	return tot
}

func (b *BasisFunc) Deriv(coeffs, x []float64, orders []int) float64 {
	b.init()
	if len(x) != b.Dim {
		panic(fmt.Sprintf("wrong dimension for x: want %v, got %v", b.Dim, len(x)))
	} else if len(orders) != b.Dim {
		panic(fmt.Sprintf("wrong number of derivative orders: want %v, got %v", b.Dim, len(orders)))
	}

	tot := 0.0
	for i, perm := range b.perms {
		cum := coeffs[i]
		for dim, exp := range perm {
			if orders[dim] > exp {
				cum *= 0
				break
			}
			cum *= float64(factorial(exp-orders[dim], exp)) * math.Pow(x[dim], float64(exp)-float64(orders[dim]))
		}
		tot += cum
	}
	return tot
}

func factorial(low, up int) int {
	tot := 1
	for i := low; i < up; i++ {
		tot *= i + 1
	}
	return tot
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

func Permute(maxsum int, dimensions ...int) [][]int {
	return permute(maxsum, dimensions, make([]int, 0, len(dimensions)))
}

func sum(vals ...int) int {
	tot := 0
	for _, val := range vals {
		tot += val
	}
	return tot
}

func permute(maxsum int, dimensions []int, prefix []int) [][]int {
	set := make([][]int, 0)

	if maxsum > 0 && sum(prefix...) >= maxsum {
		set = [][]int{append(append([]int{}, prefix...), make([]int, len(dimensions))...)}
		return set
	}

	if len(dimensions) == 1 {
		for i := 0; i < dimensions[0]; i++ {
			val := append(append([]int{}, prefix...), i)
			if maxsum == 0 || sum(val...) <= maxsum {
				set = append(set, val)
			}
		}
		return set
	}
	max := dimensions[0]
	for i := 0; i < max; i++ {
		newprefix := append(prefix, i)
		moresets := permute(maxsum, dimensions[1:], newprefix)
		set = append(set, moresets...)
	}
	return set
}
