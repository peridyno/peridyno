#include "MarchingCubesHelper.h"

#include "Topology/EdgeSet.h"

#include <thrust/sort.h>

namespace dyno
{
	__device__
	uint edgeTable[256] =
	{
		0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
		0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
		0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
		0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
		0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
		0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
		0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
		0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
		0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
		0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
		0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
		0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
		0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
		0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
		0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
		0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
		0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
		0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
		0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
		0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
		0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
		0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
		0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
		0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
		0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
		0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
		0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
		0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
		0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
		0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
		0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
		0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
	};

	__device__
	int triTable[256][16] =
	{
		{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
		{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
		{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
		{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
		{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
		{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
		{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
		{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
		{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
		{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
		{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
		{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
		{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
		{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
		{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
		{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
		{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
		{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
		{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
		{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
		{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
		{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
		{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
		{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
		{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
		{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
		{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
		{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
		{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
		{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
		{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
		{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
		{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
		{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
		{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
		{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
		{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
		{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
		{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
		{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
		{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
		{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
		{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
		{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
		{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
		{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
		{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
		{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
		{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
		{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
		{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
		{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
		{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
		{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
		{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
		{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
		{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
		{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
		{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
		{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
		{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
		{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
		{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
		{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
		{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
		{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
		{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
		{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
		{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
		{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
		{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
		{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
		{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
		{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
		{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
		{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
		{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
		{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
		{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
		{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
		{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
		{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
		{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
		{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
		{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
		{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
		{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
		{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
		{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
		{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
		{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
		{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
		{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
		{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
		{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
		{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
		{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
		{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
		{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
		{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
		{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
		{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
		{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
		{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
		{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
		{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
		{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
		{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
		{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
		{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
		{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
		{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
		{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
		{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
		{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
		{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
		{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
		{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
		{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
		{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
		{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
		{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
		{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
		{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
		{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
		{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
		{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
		{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
		{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
		{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
		{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
		{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
		{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
		{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
		{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
		{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
		{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
		{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
		{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
		{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
		{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
		{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
		{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
		{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
		{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
		{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
		{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
		{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
		{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
		{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
		{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
		{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
		{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
		{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
		{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
		{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
		{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
		{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
		{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
		{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
		{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
		{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
		{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
		{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
		{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
		{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
		{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
		{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
		{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
	};

	__device__
	uint numVertsTable[256] =
	{
		0,
		3,
		3,
		6,
		3,
		6,
		6,
		9,
		3,
		6,
		6,
		9,
		6,
		9,
		9,
		6,
		3,
		6,
		6,
		9,
		6,
		9,
		9,
		12,
		6,
		9,
		9,
		12,
		9,
		12,
		12,
		9,
		3,
		6,
		6,
		9,
		6,
		9,
		9,
		12,
		6,
		9,
		9,
		12,
		9,
		12,
		12,
		9,
		6,
		9,
		9,
		6,
		9,
		12,
		12,
		9,
		9,
		12,
		12,
		9,
		12,
		15,
		15,
		6,
		3,
		6,
		6,
		9,
		6,
		9,
		9,
		12,
		6,
		9,
		9,
		12,
		9,
		12,
		12,
		9,
		6,
		9,
		9,
		12,
		9,
		12,
		12,
		15,
		9,
		12,
		12,
		15,
		12,
		15,
		15,
		12,
		6,
		9,
		9,
		12,
		9,
		12,
		6,
		9,
		9,
		12,
		12,
		15,
		12,
		15,
		9,
		6,
		9,
		12,
		12,
		9,
		12,
		15,
		9,
		6,
		12,
		15,
		15,
		12,
		15,
		6,
		12,
		3,
		3,
		6,
		6,
		9,
		6,
		9,
		9,
		12,
		6,
		9,
		9,
		12,
		9,
		12,
		12,
		9,
		6,
		9,
		9,
		12,
		9,
		12,
		12,
		15,
		9,
		6,
		12,
		9,
		12,
		9,
		15,
		6,
		6,
		9,
		9,
		12,
		9,
		12,
		12,
		15,
		9,
		12,
		12,
		15,
		12,
		15,
		15,
		12,
		9,
		12,
		12,
		9,
		12,
		15,
		15,
		12,
		12,
		9,
		15,
		6,
		15,
		12,
		6,
		3,
		6,
		9,
		9,
		12,
		9,
		12,
		12,
		15,
		9,
		12,
		12,
		15,
		6,
		9,
		9,
		6,
		9,
		12,
		12,
		15,
		12,
		15,
		15,
		6,
		12,
		9,
		15,
		12,
		9,
		6,
		12,
		3,
		9,
		12,
		12,
		15,
		12,
		15,
		9,
		12,
		12,
		15,
		15,
		6,
		9,
		12,
		6,
		3,
		6,
		9,
		9,
		6,
		9,
		12,
		6,
		3,
		9,
		6,
		12,
		3,
		6,
		3,
		3,
		0,
	};

	template<typename Real, typename Coord, typename TDataType>
	__global__ void RconstructSDF(
		DArray3D<Real> distances,
		Coord origin,
		Real h,
		DistanceField3D<TDataType> sdf)
	{
		uint i = threadIdx.x + blockDim.x * blockIdx.x;
		uint j = threadIdx.y + blockDim.y * blockIdx.y;
		uint k = threadIdx.z + blockDim.z * blockIdx.z;

		uint nx = distances.nx();
		uint ny = distances.ny();
		uint nz = distances.nz();

		if (i >= nx || j >= ny || k >= nz) return;
		
		Coord p = origin + Coord(i * h, j * h, k * h);

		Real d;
		Coord normal;
		sdf.getDistance(p, d, normal);
		distances(i, j, k) = d;
	}

	template<typename TDataType>
	void MarchingCubesHelper<TDataType>::reconstructSDF(
		DArray3D<Real>& distances, 
		Coord origin,
		Real h, 
		DistanceField3D<TDataType>& sdf)
	{
		cuExecute3D(make_uint3(distances.nx(), distances.ny(), distances.nz()),
			RconstructSDF,
			distances,
			origin,
			h,
			sdf);
	}

	template<typename Real>
	__global__ void CountVerticeNumber(
		DArray<int> num,
		DArray3D<Real> distances,
		Real isoValue)
	{
		uint i = threadIdx.x + blockDim.x * blockIdx.x;
		uint j = threadIdx.y + blockDim.y * blockIdx.y;
		uint k = threadIdx.z + blockDim.z * blockIdx.z;

		uint nx = distances.nx();
		uint ny = distances.ny();
		uint nz = distances.nz();

		if (i >= nx - 1 || j >= ny - 1 || k >= nz - 1) return;

		uint cubeindex;
		cubeindex = uint(distances(i, j, k) < isoValue);
		cubeindex += uint(distances(i + 1, j, k) < isoValue) * 2;
		cubeindex += uint(distances(i + 1, j + 1, k) < isoValue) * 4;
		cubeindex += uint(distances(i, j + 1, k) < isoValue) * 8;
		cubeindex += uint(distances(i, j, k + 1) < isoValue) * 16;
		cubeindex += uint(distances(i + 1, j , k + 1) < isoValue) * 32;
		cubeindex += uint(distances(i + 1, j + 1, k + 1) < isoValue) * 64;
		cubeindex += uint(distances(i, j + 1, k + 1) < isoValue) * 128;

		num[i + j * (nx - 1) + k * (nx - 1) * (ny - 1)] = numVertsTable[cubeindex];
	}

	template<typename TDataType>
	void MarchingCubesHelper<TDataType>::countVerticeNumber(
		DArray<int>& num, 
		DArray3D<Real>& distances, 
		Real isoValue)
	{
		cuExecute3D(make_uint3(distances.nx() - 1, distances.ny() - 1, distances.nz() - 1),
			CountVerticeNumber,
			num,
			distances,
			isoValue);
	}

	// compute interpolated vertex along an edge
	template<typename Real, typename Coord>
	__device__	Coord vertexInterp(Real isolevel, Coord p0, Coord p1, Real f0, Real f1)
	{
		if (abs(f1 - f0) < EPSILON)
			return 0.5 * (p0 + p1);

		Real t = (isolevel - f0) / (f1 - f0);

		t = clamp(t, Real(0), Real(1));

		return (1 - t) * p0 + t * p1;
	}

	// compute interpolated vertex along an edge
	template<typename Real, typename Coord>
	__device__	Coord vertexInterp(Real& value, Coord p0, Coord p1, Real f0, Real f1, Real d0, Real d1)
	{
		if (abs(d0 - d1) < EPSILON)
		{
			value = 0.5 * (f0 + f1);
			return 0.5 * (p0 + p1);
		}
			

		Real t = d0 / (d0 - d1);

		t = clamp(t, Real(0), Real(1));

		value = (1 - t) * f0 + t * f1;

		return (1 - t) * p0 + t * p1;
	}

	template<typename Coord, typename Triangle>
	__global__ void ConstructTriangles(
		DArray<Coord> vertices,
		DArray<EKey> edgeIds,
		DArray<Triangle> triangles,
		DArray<int> vertNum,
		DArray3D<Real> distances,
		Coord origin,
		Real isoValue,
		Real h)
	{
		uint i = threadIdx.x + blockDim.x * blockIdx.x;
		uint j = threadIdx.y + blockDim.y * blockIdx.y;
		uint k = threadIdx.z + blockDim.z * blockIdx.z;

		uint nx = distances.nx();
		uint ny = distances.ny();
		uint nz = distances.nz();

		if (i >= nx - 1 || j >= ny - 1 || k >= nz - 1) return;

		Coord v[8];
		Coord p = origin + h * Coord(i, j, k);
		v[0] = p;
		v[1] = p + Coord(h, 0, 0);
		v[2] = p + Coord(h, h, 0);
		v[3] = p + Coord(0, h, 0);
		v[4] = p + Coord(0, 0, h);
		v[5] = p + Coord(h, 0, h);
		v[6] = p + Coord(h, h, h);
		v[7] = p + Coord(0, h, h);

		Real field[8];
		field[0] = distances(i, j, k);
		field[1] = distances(i + 1, j, k);
		field[2] = distances(i + 1, j + 1, k);
		field[3] = distances(i, j + 1, k);
		field[4] = distances(i, j, k + 1);
		field[5] = distances(i + 1, j, k + 1);
		field[6] = distances(i + 1, j + 1, k + 1);
		field[7] = distances(i, j + 1, k + 1);

		uint vIds[8];
		vIds[0] = distances.index(i, j, k);
		vIds[1] = distances.index(i + 1, j, k);
		vIds[2] = distances.index(i + 1, j + 1, k);
		vIds[3] = distances.index(i, j + 1, k);
		vIds[4] = distances.index(i, j, k + 1);
		vIds[5] = distances.index(i + 1, j, k + 1);
		vIds[6] = distances.index(i + 1, j + 1, k + 1);
		vIds[7] = distances.index(i, j + 1, k + 1);

		Coord vertlist[12];
		vertlist[0] = vertexInterp(isoValue, v[0], v[1], field[0], field[1]);
		vertlist[1] = vertexInterp(isoValue, v[1], v[2], field[1], field[2]);
		vertlist[2] = vertexInterp(isoValue, v[2], v[3], field[2], field[3]);
		vertlist[3] = vertexInterp(isoValue, v[3], v[0], field[3], field[0]);

		vertlist[4] = vertexInterp(isoValue, v[4], v[5], field[4], field[5]);
		vertlist[5] = vertexInterp(isoValue, v[5], v[6], field[5], field[6]);
		vertlist[6] = vertexInterp(isoValue, v[6], v[7], field[6], field[7]);
		vertlist[7] = vertexInterp(isoValue, v[7], v[4], field[7], field[4]);

		vertlist[8] = vertexInterp(isoValue, v[0], v[4], field[0], field[4]);
		vertlist[9] = vertexInterp(isoValue, v[1], v[5], field[1], field[5]);
		vertlist[10] = vertexInterp(isoValue, v[2], v[6], field[2], field[6]);
		vertlist[11] = vertexInterp(isoValue, v[3], v[7], field[3], field[7]);

		EKey edgelist[12];
		edgelist[0] = EKey(vIds[0], vIds[1]);
		edgelist[1] = EKey(vIds[1], vIds[2]);
		edgelist[2] = EKey(vIds[2], vIds[3]);
		edgelist[3] = EKey(vIds[3], vIds[0]);

		edgelist[4] = EKey(vIds[4], vIds[5]);
		edgelist[5] = EKey(vIds[5], vIds[6]);
		edgelist[6] = EKey(vIds[6], vIds[7]);
		edgelist[7] = EKey(vIds[7], vIds[4]);

		edgelist[8] = EKey(vIds[0], vIds[4]);
		edgelist[9] = EKey(vIds[1], vIds[5]);
		edgelist[10] = EKey(vIds[2], vIds[6]);
		edgelist[11] = EKey(vIds[3], vIds[7]);

		uint cubeindex;
		cubeindex = uint(distances(i, j, k) < isoValue);
		cubeindex += uint(distances(i + 1, j, k) < isoValue) * 2;
		cubeindex += uint(distances(i + 1, j + 1, k) < isoValue) * 4;
		cubeindex += uint(distances(i, j + 1, k) < isoValue) * 8;
		cubeindex += uint(distances(i, j, k + 1) < isoValue) * 16;
		cubeindex += uint(distances(i + 1, j, k + 1) < isoValue) * 32;
		cubeindex += uint(distances(i + 1, j + 1, k + 1) < isoValue) * 64;
		cubeindex += uint(distances(i, j + 1, k + 1) < isoValue) * 128;

		uint numVerts = numVertsTable[cubeindex];

		int index1D = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);

		int radix = vertNum[index1D];

		EKey e[3];
		for (int n = 0; n < numVerts; n += 3)
		{
			uint edge;
			edge = triTable[cubeindex][n];
			v[0] = vertlist[edge];
			e[0] = edgelist[edge];

			edge = triTable[cubeindex][n + 1];
			v[1] = vertlist[edge];
			e[1] = edgelist[edge];

			edge = triTable[cubeindex][n + 2];
			v[2] = vertlist[edge];
			e[2] = edgelist[edge];

			triangles[radix / 3] = Triangle(radix, radix + 1, radix + 2);

			vertices[radix] = v[0];	edgeIds[radix++] = e[0];
			vertices[radix] = v[1];	edgeIds[radix++] = e[1];
			vertices[radix] = v[2];	edgeIds[radix++] = e[2];
		}
	}

	__global__ void MCH_InitIndex(
		DArray<uint> index)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= index.size()) return;

		index[tId] = tId;
	}

	template<typename EKey>
	__global__ void MCH_CountEdgeKeys(
		DArray<int> counter,
		DArray<EKey> keys)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= keys.size()) return;

		if (tId == keys.size() - 1 || keys[tId] != keys[tId + 1])
			counter[tId] = 1;
		else
			counter[tId] = 0;
	}

	template<typename Triangle>
	__global__ void MCH_SetupNewTriangleIndex(
		DArray<Triangle> triangles,
		DArray<EKey> eKeys,
		DArray<int> radix,
		DArray<uint> ids,
		DArray<uint> rIds)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= triangles.size()) return;

		Triangle t = triangles[tId];

		uint v0 = rIds[t[0]];
		uint v1 = rIds[t[1]];
		uint v2 = rIds[t[2]];

		triangles[tId] = Triangle(v0, v1, v2);
	}

	template<typename Coord>
	__global__ void MCH_SetupNewVertices(
		DArray<Coord> newVertices,
		DArray<Coord> vertices,
		DArray<uint> rIds,
		DArray<EKey> keys,
		DArray<int> radix,
		DArray<uint> ids)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= vertices.size()) return;

		if (tId == vertices.size() - 1 || keys[tId] != keys[tId + 1]) {
			newVertices[radix[tId]] = vertices[ids[tId]];
		}

		rIds[ids[tId]] = radix[tId];

//		printf("%d %d \n", tId, ids[tId]);
	}

	template<typename TDataType>
	void MarchingCubesHelper<TDataType>::constructTriangles(
		DArray<Coord>& vertices,
		DArray<TopologyModule::Triangle>& triangles,
		DArray<int>& vertNum, 
		DArray3D<Real>& distances,
		Coord origin,
		Real isoValue, 
		Real h)
	{
		DArray<EKey> eKeys(vertices.size());
		DArray<uint> ids(vertices.size());
		DArray<int> counter(vertices.size());
		DArray<uint> rIds(vertices.size());

		cuExecute3D(make_uint3(distances.nx() - 1, distances.ny() - 1, distances.nz() - 1),
			ConstructTriangles,
			vertices,
			eKeys,
			triangles,
			vertNum,
			distances,
			origin,
			isoValue,
			h);

		cuExecute(ids.size(),
			MCH_InitIndex,
			ids);

		thrust::sort_by_key(thrust::device, eKeys.begin(), eKeys.begin() + eKeys.size(), ids.begin());

		cuExecute(eKeys.size(),
			MCH_CountEdgeKeys,
			counter,
			eKeys);

		int num = thrust::reduce(thrust::device, counter.begin(), counter.begin() + counter.size());
		thrust::exclusive_scan(thrust::device, counter.begin(), counter.begin() + counter.size(), counter.begin());

		DArray<Coord> newVertices(num);

		cuExecute(vertices.size(),
			MCH_SetupNewVertices,
			newVertices,
			vertices,
			rIds,
			eKeys,
			counter,
			ids);

		cuExecute(triangles.size(),
			MCH_SetupNewTriangleIndex,
			triangles,
			eKeys,
			counter,
			ids,
			rIds);

		vertices.assign(newVertices);

		ids.clear();
		rIds.clear();
		eKeys.clear();
		counter.clear();
		newVertices.clear();
	}

	template<typename Real, typename Coord>
	__global__ void MCH_CountVertexNumberForClipper(
		DArray<int> num,
		DArray3D<Real> distances,
		Coord origin,
		Coord h,
		TPlane3D<Real> plane)
	{
		uint i = threadIdx.x + blockDim.x * blockIdx.x;
		uint j = threadIdx.y + blockDim.y * blockIdx.y;
		uint k = threadIdx.z + blockDim.z * blockIdx.z;

		uint nx = distances.nx();
		uint ny = distances.ny();
		uint nz = distances.nz();

		if (i >= nx - 1 || j >= ny - 1 || k >= nz - 1) return;

		Real d;
		TPoint3D<Real> p;

		Real isoValue = 0.0;

		uint cubeindex;

		p = origin + Coord(i * h[0], j * h[1], k * h[2]);
		d = p.distance(plane);
		cubeindex = uint(d < isoValue);

		p = origin + Coord((i + 1) * h[0], j * h[1], k * h[2]);
		d = p.distance(plane);
		cubeindex += uint(d < isoValue) * 2;

		p = origin + Coord((i + 1) * h[0], (j + 1) * h[1], k * h[2]);
		d = p.distance(plane);
		cubeindex += uint(d < isoValue) * 4;

		p = origin + Coord(i * h[0], (j + 1) * h[1], k * h[2]);
		d = p.distance(plane);
		cubeindex += uint(d < isoValue) * 8;

		p = origin + Coord(i * h[0], j * h[1], (k + 1) * h[2]);
		d = p.distance(plane);
		cubeindex += uint(d < isoValue) * 16;

		p = origin + Coord((i + 1) * h[0], j * h[1], (k + 1) * h[2]);
		d = p.distance(plane);
		cubeindex += uint(d < isoValue) * 32;

		p = origin + Coord((i + 1) * h[0], (j + 1) * h[1], (k + 1) * h[2]);
		d = p.distance(plane);
		cubeindex += uint(d < isoValue) * 64;

		p = origin + Coord(i * h[0], (j + 1) * h[1], (k + 1) * h[2]);
		d = p.distance(plane);
		cubeindex += uint(d < isoValue) * 128;


		num[i + j * (nx - 1) + k * (nx - 1) * (ny - 1)] = numVertsTable[cubeindex];
	}

	template<typename TDataType>
	void MarchingCubesHelper<TDataType>::countVerticeNumberForClipper(DArray<int>& num, DistanceField3D<TDataType>& sdf, TPlane3D<Real> plane)
	{
		auto& distances = sdf.getMDistance();

		cuExecute3D(make_uint3(distances.nx() - 1, distances.ny() - 1, distances.nz() - 1),
			MCH_CountVertexNumberForClipper,
			num,
			distances,
			sdf.lowerBound(),
			sdf.getH(),
			plane);
	}

	template<typename Real, typename Coord, typename Triangle>
	__global__ void MCH_ConstructTrianglesForClipper(
		DArray<Real> sdfValues,
		DArray<Coord> vertices,
		DArray<Triangle> triangles,
		DArray<int> vertRadix,
		DArray3D<Real> distances,
		Coord origin,
		Coord h,
		TPlane3D<Real> plane) 
	{
		uint i = threadIdx.x + blockDim.x * blockIdx.x;
		uint j = threadIdx.y + blockDim.y * blockIdx.y;
		uint k = threadIdx.z + blockDim.z * blockIdx.z;

		uint nx = distances.nx();
		uint ny = distances.ny();
		uint nz = distances.nz();

		if (i >= nx - 1 || j >= ny - 1 || k >= nz - 1) return;

		Coord v[8];
		Coord p = origin + Coord(i * h[0], j * h[1], k * h[2]);
		v[0] = p;
		v[1] = p + Coord(h[0], 0, 0);
		v[2] = p + Coord(h[0], h[1], 0);
		v[3] = p + Coord(0, h[1], 0);
		v[4] = p + Coord(0, 0, h[2]);
		v[5] = p + Coord(h[0], 0, h[2]);
		v[6] = p + Coord(h[0], h[1], h[2]);
		v[7] = p + Coord(0, h[1], h[2]);

		Real field[8];
		field[0] = distances(i, j, k);
		field[1] = distances(i + 1, j, k);
		field[2] = distances(i + 1, j + 1, k);
		field[3] = distances(i, j + 1, k);
		field[4] = distances(i, j, k + 1);
		field[5] = distances(i + 1, j, k + 1);
		field[6] = distances(i + 1, j + 1, k + 1);
		field[7] = distances(i, j + 1, k + 1);

		Real isoValue = 0.0;

		uint cubeindex;

		TPoint3D<Real> pos;
		pos = v[0];
		Real d0 = pos.distance(plane);
		cubeindex = uint(d0 < isoValue);

		pos = v[1];
		Real d1 = pos.distance(plane);
		cubeindex += uint(d1 < isoValue) * 2;

		pos = v[2];
		Real d2 = pos.distance(plane);
		cubeindex += uint(d2 < isoValue) * 4;

		pos = v[3];
		Real d3 = pos.distance(plane);
		cubeindex += uint(d3 < isoValue) * 8;

		pos = v[4];
		Real d4 = pos.distance(plane);
		cubeindex += uint(d4 < isoValue) * 16;

		pos = v[5];
		Real d5 = pos.distance(plane);
		cubeindex += uint(d5 < isoValue) * 32;

		pos = v[6];
		Real d6 = pos.distance(plane);
		cubeindex += uint(d6 < isoValue) * 64;

		pos = v[7];
		Real d7 = pos.distance(plane);
		cubeindex += uint(d7 < isoValue) * 128;

		Real scalar[12];
		Coord vertlist[12];
		vertlist[0] = vertexInterp(scalar[0], v[0], v[1], field[0], field[1], d0, d1);
		vertlist[1] = vertexInterp(scalar[1], v[1], v[2], field[1], field[2], d1, d2);
		vertlist[2] = vertexInterp(scalar[2], v[2], v[3], field[2], field[3], d2, d3);
		vertlist[3] = vertexInterp(scalar[3], v[3], v[0], field[3], field[0], d3, d0);

		vertlist[4] = vertexInterp(scalar[4], v[4], v[5], field[4], field[5], d4, d5);
		vertlist[5] = vertexInterp(scalar[5], v[5], v[6], field[5], field[6], d5, d6);
		vertlist[6] = vertexInterp(scalar[6], v[6], v[7], field[6], field[7], d6, d7);
		vertlist[7] = vertexInterp(scalar[7], v[7], v[4], field[7], field[4], d7, d4);

		vertlist[8] = vertexInterp(scalar[8], v[0], v[4], field[0], field[4], d0, d4);
		vertlist[9] = vertexInterp(scalar[9], v[1], v[5], field[1], field[5], d1, d5);
		vertlist[10] = vertexInterp(scalar[10], v[2], v[6], field[2], field[6], d2, d6);
		vertlist[11] = vertexInterp(scalar[10], v[3], v[7], field[3], field[7], d3, d7);

		int index1D = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);

		int radix = vertRadix[index1D];

		uint numVerts = numVertsTable[cubeindex];

		Real c[3];
		for (int n = 0; n < numVerts; n += 3)
		{
			uint edge;
			edge = triTable[cubeindex][n];
			v[0] = vertlist[edge];
			c[0] = scalar[edge];

			edge = triTable[cubeindex][n + 1];
			v[1] = vertlist[edge];
			c[1] = scalar[edge];

			edge = triTable[cubeindex][n + 2];
			v[2] = vertlist[edge];
			c[2] = scalar[edge];

			triangles[radix / 3] = Triangle(radix, radix + 1, radix + 2);

			vertices[radix] = v[0];	sdfValues[radix] = c[0];	radix++;
			vertices[radix] = v[1];	sdfValues[radix] = c[1];	radix++;
			vertices[radix] = v[2];	sdfValues[radix] = c[2];	radix++;
		}
	}

	template<typename TDataType>
	void MarchingCubesHelper<TDataType>::constructTrianglesForClipper(
		DArray<Real>& field,
		DArray<Coord>& vertices, 
		DArray<TopologyModule::Triangle>& triangles, 
		DArray<int>& vertNum, 
		DistanceField3D<TDataType>& sdf, 
		TPlane3D<Real> plane)
	{
		auto& distances = sdf.getMDistance();

		cuExecute3D(make_uint3(distances.nx() - 1, distances.ny() - 1, distances.nz() - 1),
			MCH_ConstructTrianglesForClipper,
			field,
			vertices,
			triangles,
			vertNum,
			distances,
			sdf.lowerBound(),
			sdf.getH(),
			plane);
	}

	template<typename Real, typename Coord>
	__global__ void CountVerticeNumberForOctree(
		DArray<uint> num,
		DArray<Coord> vertices,
		DArray<Real> sdfs,
		Real isoValue)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= num.size()) return;

		int vIdx = 8 * pId;

		uint cubeindex;
		cubeindex = uint(sdfs[vIdx] < isoValue);
		cubeindex += uint(sdfs[vIdx + 1] < isoValue) * 2;
		cubeindex += uint(sdfs[vIdx + 2] < isoValue) * 4;
		cubeindex += uint(sdfs[vIdx + 3] < isoValue) * 8;
		cubeindex += uint(sdfs[vIdx + 4] < isoValue) * 16;
		cubeindex += uint(sdfs[vIdx + 5] < isoValue) * 32;
		cubeindex += uint(sdfs[vIdx + 6] < isoValue) * 64;
		cubeindex += uint(sdfs[vIdx + 7] < isoValue) * 128;

		num[pId] = numVertsTable[cubeindex];
	}

	template<typename TDataType>
	void MarchingCubesHelper<TDataType>::countVerticeNumberForOctree(
		DArray<uint>& num, 
		DArray<Coord>& vertices, 
		DArray<Real>& sdfs, 
		Real isoValue)
	{
		cuExecute(num.size(),
			CountVerticeNumberForOctree,
			num,
			vertices,
			sdfs,
			isoValue);
	}

	template<typename Real, typename Coord>
	__global__ void ConstructTrianglesForOctree(
		DArray<Coord> triangleVertices,
		DArray<TopologyModule::Triangle> triangles,
		DArray<uint> vertNum,
		DArray<Coord> cellVertices,
		DArray<Real> sdfs,
		Real isoValue)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vertNum.size()) return;

		int vIdx = 8 * pId;

		Coord v[8];
		v[0] = cellVertices[vIdx];
		v[1] = cellVertices[vIdx + 1];
		v[2] = cellVertices[vIdx + 2];
		v[3] = cellVertices[vIdx + 3];
		v[4] = cellVertices[vIdx + 4];
		v[5] = cellVertices[vIdx + 5];
		v[6] = cellVertices[vIdx + 6];
		v[7] = cellVertices[vIdx + 7];

		Real field[8];
		field[0] = sdfs[vIdx];
		field[1] = sdfs[vIdx + 1];
		field[2] = sdfs[vIdx + 2];
		field[3] = sdfs[vIdx + 3];
		field[4] = sdfs[vIdx + 4];
		field[5] = sdfs[vIdx + 5];
		field[6] = sdfs[vIdx + 6];
		field[7] = sdfs[vIdx + 7];

		Coord vertlist[12];
		vertlist[0] = vertexInterp(isoValue, v[0], v[1], field[0], field[1]);
		vertlist[1] = vertexInterp(isoValue, v[1], v[2], field[1], field[2]);
		vertlist[2] = vertexInterp(isoValue, v[2], v[3], field[2], field[3]);
		vertlist[3] = vertexInterp(isoValue, v[3], v[0], field[3], field[0]);

		vertlist[4] = vertexInterp(isoValue, v[4], v[5], field[4], field[5]);
		vertlist[5] = vertexInterp(isoValue, v[5], v[6], field[5], field[6]);
		vertlist[6] = vertexInterp(isoValue, v[6], v[7], field[6], field[7]);
		vertlist[7] = vertexInterp(isoValue, v[7], v[4], field[7], field[4]);

		vertlist[8] = vertexInterp(isoValue, v[0], v[4], field[0], field[4]);
		vertlist[9] = vertexInterp(isoValue, v[1], v[5], field[1], field[5]);
		vertlist[10] = vertexInterp(isoValue, v[2], v[6], field[2], field[6]);
		vertlist[11] = vertexInterp(isoValue, v[3], v[7], field[3], field[7]);

		uint cubeindex;
		cubeindex = uint(sdfs[vIdx] < isoValue);
		cubeindex += uint(sdfs[vIdx + 1] < isoValue) * 2;
		cubeindex += uint(sdfs[vIdx + 2] < isoValue) * 4;
		cubeindex += uint(sdfs[vIdx + 3] < isoValue) * 8;
		cubeindex += uint(sdfs[vIdx + 4] < isoValue) * 16;
		cubeindex += uint(sdfs[vIdx + 5] < isoValue) * 32;
		cubeindex += uint(sdfs[vIdx + 6] < isoValue) * 64;
		cubeindex += uint(sdfs[vIdx + 7] < isoValue) * 128;

		uint numVerts = numVertsTable[cubeindex];

		int radix = vertNum[pId];

		for (int n = 0; n < numVerts; n += 3)
		{
			uint edge;
			edge = triTable[cubeindex][n];
			v[0] = vertlist[edge];

			edge = triTable[cubeindex][n + 1];
			v[1] = vertlist[edge];

			edge = triTable[cubeindex][n + 2];
			v[2] = vertlist[edge];

			triangles[radix / 3] = TopologyModule::Triangle(radix, radix + 1, radix + 2);

			triangleVertices[radix++] = v[0];
			triangleVertices[radix++] = v[1];
			triangleVertices[radix++] = v[2];
		}
	}

	template<typename TDataType>
	void MarchingCubesHelper<TDataType>::constructTrianglesForOctree(
		DArray<Coord>& triangleVertices,
		DArray<TopologyModule::Triangle>& triangles,
		DArray<uint>& num,
		DArray<Coord>& cellVertices,
		DArray<Real>& sdfs,
		Real isoValue)
	{
		cuExecute(num.size(),
			ConstructTrianglesForOctree,
			triangleVertices,
			triangles,
			num,
			cellVertices,
			sdfs,
			isoValue);
	}

	template<typename Real, typename Coord>
	__global__ void MCH_CountVertexNumberForOctreeClipper(
		DArray<uint> num,
		DArray<Coord> vertices,
		TPlane3D<Real> plane)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= num.size()) return;

		int vIdx = 8 * pId;

		Real d;
		TPoint3D<Real> p;

		Real isoValue = 0.0;

		uint cubeindex;

		p = vertices[vIdx];
		d = p.distance(plane);
		cubeindex = uint(d < isoValue);

		p = vertices[vIdx + 1];
		d = p.distance(plane);
		cubeindex += uint(d < isoValue) * 2;

		p = vertices[vIdx + 2];
		d = p.distance(plane);
		cubeindex += uint(d < isoValue) * 4;

		p = vertices[vIdx + 3];
		d = p.distance(plane);
		cubeindex += uint(d < isoValue) * 8;

		p = vertices[vIdx + 4];
		d = p.distance(plane);
		cubeindex += uint(d < isoValue) * 16;

		p = vertices[vIdx + 5];
		d = p.distance(plane);
		cubeindex += uint(d < isoValue) * 32;

		p = vertices[vIdx + 6];
		d = p.distance(plane);
		cubeindex += uint(d < isoValue) * 64;

		p = vertices[vIdx + 7];
		d = p.distance(plane);
		cubeindex += uint(d < isoValue) * 128;

		num[pId] = numVertsTable[cubeindex];
	}

	template<typename TDataType>
	void MarchingCubesHelper<TDataType>::countVerticeNumberForOctreeClipper(
		DArray<uint>& num, 
		DArray<Coord>& vertices,  
		TPlane3D<Real> plane)
	{
		cuExecute(num.size(),
			MCH_CountVertexNumberForOctreeClipper,
			num,
			vertices,
			plane);
	}

	template<typename Real, typename Coord>
	__global__ void MCH_ConstructTrianglesForOctreeClipper(
		DArray<Real> vertSDFs,
		DArray<Coord> triangleVertices,
		DArray<TopologyModule::Triangle> triangles,
		DArray<uint> vertNum,
		DArray<Coord> cellVertices,
		DArray<Real> sdfs,
		TPlane3D<Real> plane)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vertNum.size()) return;

		int vIdx = 8 * pId;

		Coord v[8];
		v[0] = cellVertices[vIdx];
		v[1] = cellVertices[vIdx + 1];
		v[2] = cellVertices[vIdx + 2];
		v[3] = cellVertices[vIdx + 3];
		v[4] = cellVertices[vIdx + 4];
		v[5] = cellVertices[vIdx + 5];
		v[6] = cellVertices[vIdx + 6];
		v[7] = cellVertices[vIdx + 7];

		Real isoValue = Real(0);

		Real field[8];
		field[0] = sdfs[vIdx];
		field[1] = sdfs[vIdx + 1];
		field[2] = sdfs[vIdx + 2];
		field[3] = sdfs[vIdx + 3];
		field[4] = sdfs[vIdx + 4];
		field[5] = sdfs[vIdx + 5];
		field[6] = sdfs[vIdx + 6];
		field[7] = sdfs[vIdx + 7];

		uint cubeindex;
		TPoint3D<Real> pos;
		pos = cellVertices[vIdx];
		Real d0 = pos.distance(plane);
		cubeindex = uint(d0 < isoValue);

		pos = cellVertices[vIdx + 1];
		Real d1 = pos.distance(plane);
		cubeindex += uint(d1 < isoValue) * 2;

		pos = cellVertices[vIdx + 2];
		Real d2 = pos.distance(plane);
		cubeindex += uint(d2 < isoValue) * 4;

		pos = cellVertices[vIdx + 3];
		Real d3 = pos.distance(plane);
		cubeindex += uint(d3 < isoValue) * 8;

		pos = cellVertices[vIdx + 4];
		Real d4 = pos.distance(plane);
		cubeindex += uint(d4 < isoValue) * 16;

		pos = cellVertices[vIdx + 5];
		Real d5 = pos.distance(plane);
		cubeindex += uint(d5 < isoValue) * 32;

		pos = cellVertices[vIdx + 6];
		Real d6 = pos.distance(plane);
		cubeindex += uint(d6 < isoValue) * 64;

		pos = cellVertices[vIdx + 7];
		Real d7 = pos.distance(plane);
		cubeindex += uint(d7 < isoValue) * 128;

		Real scalar[12];
		Coord vertlist[12];
		vertlist[0] = vertexInterp(scalar[0], v[0], v[1], field[0], field[1], d0, d1);
		vertlist[1] = vertexInterp(scalar[1], v[1], v[2], field[1], field[2], d1, d2);
		vertlist[2] = vertexInterp(scalar[2], v[2], v[3], field[2], field[3], d2, d3);
		vertlist[3] = vertexInterp(scalar[3], v[3], v[0], field[3], field[0], d3, d0);

		vertlist[4] = vertexInterp(scalar[4], v[4], v[5], field[4], field[5], d4, d5);
		vertlist[5] = vertexInterp(scalar[5], v[5], v[6], field[5], field[6], d5, d6);
		vertlist[6] = vertexInterp(scalar[6], v[6], v[7], field[6], field[7], d6, d7);
		vertlist[7] = vertexInterp(scalar[7], v[7], v[4], field[7], field[4], d7, d4);

		vertlist[8] = vertexInterp(scalar[8], v[0], v[4], field[0], field[4], d0, d4);
		vertlist[9] = vertexInterp(scalar[9], v[1], v[5], field[1], field[5], d1, d5);
		vertlist[10] = vertexInterp(scalar[10], v[2], v[6], field[2], field[6], d2, d6);
		vertlist[11] = vertexInterp(scalar[10], v[3], v[7], field[3], field[7], d3, d7);

		uint numVerts = numVertsTable[cubeindex];

		int radix = vertNum[pId];

		Real c[3];
		for (int n = 0; n < numVerts; n += 3)
		{
			uint edge;
			edge = triTable[cubeindex][n];
			v[0] = vertlist[edge];
			c[0] = scalar[edge];

			edge = triTable[cubeindex][n + 1];
			v[1] = vertlist[edge];
			c[1] = scalar[edge];

			edge = triTable[cubeindex][n + 2];
			v[2] = vertlist[edge];
			c[2] = scalar[edge];

			triangles[radix / 3] = TopologyModule::Triangle(radix, radix + 1, radix + 2);

			triangleVertices[radix] = v[0];	vertSDFs[radix] = c[0];	radix++;
			triangleVertices[radix] = v[1];	vertSDFs[radix] = c[1];	radix++;
			triangleVertices[radix] = v[2];	vertSDFs[radix] = c[2];	radix++;
		}
	}

	template<typename TDataType>
	void MarchingCubesHelper<TDataType>::constructTrianglesForOctreeClipper(
		DArray<Real>& vertSDFs,
		DArray<Coord>& triangleVertices, 
		DArray<TopologyModule::Triangle>& triangles, 
		DArray<uint>& num, 
		DArray<Coord>& cellVertices, 
		DArray<Real>& sdfs,
		TPlane3D<Real> plane)
	{
		cuExecute(num.size(),
			MCH_ConstructTrianglesForOctreeClipper,
			vertSDFs,
			triangleVertices,
			triangles,
			num,
			cellVertices,
			sdfs,
			plane);
	}

	DEFINE_CLASS(MarchingCubesHelper);
}