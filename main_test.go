package main

import (
	"fmt"
	"testing"
)

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
