package main

import "sort"

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

func L2DistSquared(a, b []float64) float64 {
	tot := 0.0
	for i := range a {
		diff := a[i] - b[i]
		tot += diff * diff
	}
	return tot
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

	if maxsum > 0 {
		if tot := sum(prefix...); tot == maxsum {
			return [][]int{append(append([]int{}, prefix...), make([]int, len(dimensions))...)}
		} else if tot > maxsum {
			return set
		}
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

func factorial(low, up int) int {
	tot := 1
	for i := low; i < up; i++ {
		tot *= i + 1
	}
	return tot
}
