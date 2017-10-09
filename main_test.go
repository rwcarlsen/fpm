package main

import (
	"fmt"
	"math"
	"testing"
)

func TestBasisFunc_TermsAtZero(t *testing.T) {
	var tests = []struct {
		dims        int
		degree      int
		derivOrders []int
		want        float64
	}{
		{dims: 2, degree: 1, derivOrders: []int{0, 0}, want: 1},
		{dims: 2, degree: 1, derivOrders: []int{1, 0}, want: 1},
		{dims: 2, degree: 1, derivOrders: []int{0, 1}, want: 1},
		{dims: 2, degree: 1, derivOrders: []int{1, 1}, want: 0},
		{dims: 2, degree: 2, derivOrders: []int{0, 0}, want: 1},
		{dims: 2, degree: 2, derivOrders: []int{1, 0}, want: 1},
		{dims: 2, degree: 2, derivOrders: []int{0, 1}, want: 1},
		{dims: 2, degree: 2, derivOrders: []int{1, 1}, want: 1},
		{dims: 2, degree: 2, derivOrders: []int{2, 0}, want: 2},
		{dims: 2, degree: 2, derivOrders: []int{0, 2}, want: 2},
		{dims: 2, degree: 3, derivOrders: []int{1, 2}, want: 2},
		{dims: 2, degree: 3, derivOrders: []int{2, 1}, want: 2},
		{dims: 2, degree: 3, derivOrders: []int{3, 0}, want: 6},
		{dims: 2, degree: 3, derivOrders: []int{0, 3}, want: 6},
	}

	const tol = 1e-6
	for i, test := range tests {
		bf := &BasisFunc{Dim: test.dims, Degree: test.degree}
		_, got := bf.TermsAtZero(test.derivOrders)
		t.Run(fmt.Sprintf("case%v", i+1), testFloat(got, test.want, tol, "dims=%v, degree=%v, derivs=%v: ", test.dims, test.degree, test.derivOrders))
	}
}

func TestBasisFunc_Val(t *testing.T) {
	var tests = []struct {
		coeffs []float64
		x      []float64
		degree int
		want   float64
	}{
		{degree: 1, coeffs: []float64{1, 1, 1}, x: []float64{0, 0}, want: 1},
		{degree: 1, coeffs: []float64{1, 1, 1}, x: []float64{1, 0}, want: 2},
		{degree: 1, coeffs: []float64{1, 1, 1}, x: []float64{0, 1}, want: 2},
		{degree: 1, coeffs: []float64{1, 1, 1}, x: []float64{1, 1}, want: 3},
	}

	const tol = 1e-6
	for i, test := range tests {
		bf := &BasisFunc{Dim: len(test.x), Degree: test.degree}
		got := bf.Val(test.coeffs, test.x)
		t.Run(fmt.Sprintf("case%v", i+1), testFloat(got, test.want, tol, "coeffs=%v, x=%v: ", test.coeffs, test.x))
	}
}

func testFloat(got, want, tol float64, msg string, args ...interface{}) func(t *testing.T) {
	return func(t *testing.T) {
		if math.Abs(got-want) > tol {
			t.Errorf(msg+"want %v, got %v  (tol=%v)", append(args, want, got, tol)...)
		}
	}
}

func TestPermute(t *testing.T) {
	var tests = []struct {
		dims   []int
		maxsum int
		want   [][]int
	}{
		{
			dims:   []int{2, 2},
			maxsum: 2,
			want: [][]int{
				{0, 0},
				{0, 1},
				{1, 0},
				{1, 1},
			},
		}, {
			dims:   []int{3, 3},
			maxsum: 2,
			want: [][]int{
				{0, 0},
				{0, 1},
				{0, 2},
				{1, 0},
				{1, 1},
				{2, 0},
			},
		}, {
			dims:   []int{3, 3},
			maxsum: 3,
			want: [][]int{
				{0, 0},
				{0, 1},
				{0, 2},
				{1, 0},
				{1, 1},
				{1, 2},
				{2, 0},
				{2, 1},
				{2, 2},
			},
		},
	}

	for i, test := range tests {
		got := Permute(test.maxsum, test.dims...)
		t.Run(fmt.Sprintf("case%v:dims=%v,maxsum=%v", i+1, test.dims, test.maxsum), testPermute(got, test.want))
	}
}

func testPermute(got, want [][]int) func(t *testing.T) {
	return func(t *testing.T) {
		for i, gotperm := range got {
			wantperm := want[i]
			for j := range gotperm {
				if gotperm[j] != wantperm[j] {
					t.Errorf("perm %v: want %v, got %v", i+1, wantperm, gotperm)
					break
				}
			}
		}
	}
}
