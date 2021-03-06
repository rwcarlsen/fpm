package main

import (
	"fmt"
	"math"
)

type WeightFunc interface {
	Weight(xref, x []float64) float64
}

type UniformWeight struct{}

func (n UniformWeight) Weight(xref, x []float64) float64 { return 1 }

// NormGauss implements a compactly supported radial basis function using the normalized Gaussian
// function.
type NormGauss struct {
	// The value of the type is the support range/distance for the neighborhood for which
	// weights are being calculated.
	Rho float64
	// Epsilon is the shape parameter
	Epsilon float64
}

func (n NormGauss) Weight(xref, x []float64) float64 {
	distsq := L2DistSquared(xref, x)
	rhosq := n.Rho * n.Rho
	if distsq >= rhosq {
		return 0
	}
	eps := n.Epsilon
	return (math.Exp(-eps*distsq/rhosq) - math.Exp(-eps)) / (1 - math.Exp(-eps))
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
	}
}

// TermAtZero calculates and returns the multiplier (due to derivaties) on the monomial that
// matches the set of derivative orders (nth derivative for each dimension/variable) and its
// associated index.  This is equivalent to the partial derivative described by derivOrders of the
// basis function evaluated at X=0 (all dimensions zero).  This panics if there is no non-zero basis
// function term for the given derivative combination; this would be indicative of your basis
// function degree (i.e. the number of interpolating monomials) being too low for the derivative
// order of your differential equation being solved.
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
	panic("highest monomial order is too low for kernel derivative order")
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
