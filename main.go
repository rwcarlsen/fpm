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
	}
	pts := [][]float64{
		{0, 0},
		{1, 2},
	}

	for i := range coeffs {
		v := bf.Val(coeffs[i], pts[i])
		fmt.Printf("basisfunc%v=%v\n", pts[i], v)
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
		b.perms = Permute(dims...)
	}
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
		fmt.Printf("    term %v=1", i)
		for dim, exp := range perm {
			fmt.Printf("*dim%v^%v", dim+1, exp)
			cum *= math.Pow(x[dim], float64(exp))
		}
		fmt.Println()
		tot += cum
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

func Permute(dimensions ...int) [][]int {
	return permute(dimensions, make([]int, 0, len(dimensions)))
}

func permute(dimensions []int, prefix []int) [][]int {
	set := make([][]int, 0)

	if len(dimensions) == 1 {
		for i := 0; i < dimensions[0]; i++ {
			val := append(append([]int{}, prefix...), i)
			set = append(set, val)
		}
		return set
	}

	max := dimensions[0]
	for i := 0; i < max; i++ {
		newprefix := append(prefix, i)
		moresets := permute(dimensions[1:], newprefix)
		set = append(set, moresets...)
	}
	return set
}
