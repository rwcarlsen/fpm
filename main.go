package main

import (
	"fmt"
	"math"

	"sort"
)

func main() {
	ps := PointSet{
		NewPoint(0, 0, 0),
		NewPoint(0, 0, 1),
		NewPoint(0, 1, 1),
		NewPoint(1, 1, 1),
	}

	p := NewPoint(0, 1, 1)

	fmt.Printf("nearest 1 to %v: %v\n", p, ps.Nearest(p, 1))
	fmt.Printf("nearest 2 to %v: %v\n", p, ps.Nearest(p, 2))
	fmt.Printf("nearest 3 to %v: %v\n", p, ps.Nearest(p, 3))

	bf := &BasisFunc{Dim: 2, Degree: 2}

	coeffs := [][]float64{
		{1, 1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1, 1},
	}
	pts := [][]float64{
		{0, 0},
		{1, 2},
		{0, 0},
		{0, 0},
		{0, 0},
	}
	orders := [][]int{
		{0, 0},
		{1, 0},
		{0, 1},
		{1, 1},
		{0, 2},
	}

	for i := range coeffs {
		v := bf.Val(coeffs[i], pts[i])
		deriv := bf.Deriv(coeffs[i], pts[i], orders[i])
		fmt.Printf("basisfunc%v=%v\n", pts[i], v)
		fmt.Printf("partial func%v wrt %v = %v\n", pts[i], orders[i], deriv)
	}
}

type BasisFunc struct {
	Dim    int
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
	}
}

// TermsAtZero calculates and returns the multiplier (due to derivaties) on the monomial that
// matches the set of derivative orders (nth derivative for each dimension/variable) and its
// associated index.  This is equivalent to the partial derivative described by derivOrders of the
// basis function evaluated at X=0 (all dimensions zero).
func (b *BasisFunc) TermsAtZero(derivOrders []int) (index int, multiplier float64) {
	b.init()
	if len(derivOrders) != b.Dim {
		panic(fmt.Sprintf("wrong number of derivative orders: want %v, got %v", b.Dim, len(derivOrders)))
	}

outer:
	for i, perm := range b.perms {
		fmt.Printf("comparing perm %v to derivOrders %v\n", perm, derivOrders)
		for dim, exp := range perm {
			if exp != derivOrders[dim] {
				continue outer
			}
		}
		mult := 1.0
		for _, exp := range perm {
			mult *= float64(factorial(0, exp))
			fmt.Printf("    perm matches, mult=%v\n", mult)
		}
		return i, mult
	}
	return -1, 0
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

type Point struct {
	X      []float64
	Coeffs []float64
}

func NewPoint(xs ...float64) *Point {
	return &Point{X: xs}
}

func (p *Point) init(x *Point) {
	if len(p.X) != len(x.X) {
		p.X = make([]float64, len(x.X))
	}
}

func (p *Point) Add(p1, p2 *Point) {
	p.init(p1)
	for i := range p.X {
		p.X[i] = p1.X[i] + p2.X[i]
	}
}

func (p *Point) Sub(p1, p2 *Point) {
	p.init(p1)
	for i := range p.X {
		p.X[i] = p2.X[i] - p1.X[i]
	}
}

func (p *Point) L2Norm() float64 {
	tot := 0.0
	for _, x := range p.X {
		tot += x * x
	}
	return math.Sqrt(tot)
}

func (p *Point) Dist(y *Point) float64 {
	var diff Point
	diff.Sub(p, y)
	return diff.L2Norm()
}

type PointSet []*Point

func (ps PointSet) LocalSolution(i int) float64 {
	panic("unimplemented")
}

func (ps PointSet) Nearest(p *Point, n int) []int {
	sorted := make([]int, len(ps))
	for i := range sorted {
		sorted[i] = i
	}
	sort.Slice(sorted, func(i, j int) bool {
		return ps[sorted[i]].Dist(p) < ps[sorted[j]].Dist(p)
	})

	nearest := make([]int, n)
	for i := range nearest {
		nearest[i] = sorted[i]
	}

	return nearest
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
