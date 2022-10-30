#include "gtest/gtest.h"
#include "Array/Array.h"
#include "Array/Array2D.h"
#include "Array/Array3D.h"
#include "Array/ArrayList.h"
#include "Array/ArrayMap.h"
#include <thrust/sort.h>
#include <vector>

using namespace dyno;

TEST(Array, CPU)
{
	CArray<int> cArr;

	cArr.pushBack(1);
	cArr.pushBack(2);

	EXPECT_EQ(cArr.size(), 2);
	std::cout << cArr;

	DArray<int> gArr;	
	gArr.assign(cArr);

	EXPECT_EQ(gArr.size(), 2);
	std::cout << gArr;

	DArrayList<int> arrList;
	arrList.resize(gArr);

	EXPECT_EQ(arrList.elementSize(), 3);

	DArrayMap<int> arrMap;
	arrMap.resize(gArr);

	EXPECT_EQ(arrMap.elementSize(), 3);

	DArrayList<int> constList;
	constList.resize(5, 2);

	EXPECT_EQ(constList.elementSize(), 10);

	DArrayMap<int> constMap;
	constMap.resize(5, 3);

	EXPECT_EQ(constMap.elementSize(), 15);
}

TEST(Array, Copy)
{
	CArray<int> cArr;
	CArray<int> cArr2;
	DArray<int> gArr;
	DArray<int> gArr2;

	cArr.pushBack(1);
	cArr.pushBack(2);

	gArr.resize(cArr.size());

	cArr2.assign(cArr);
	gArr.assign(cArr);

	cArr.assign(gArr);

	gArr2.resize(gArr.size());
//	gArr2.copyFrom(gArr);
}

TEST(Array, assign)
{
	CArray<int> cArr(6);
	cArr.assign(5);
	EXPECT_EQ(cArr.size() == 6, true);
	EXPECT_EQ(cArr[0] == 5, true);

	cArr.assign(3);
	EXPECT_EQ(cArr[0] == 3, true);

	cArr.assign(2, 1);
	EXPECT_EQ(cArr.size() == 2, true);
	EXPECT_EQ(cArr[0] == 1, true);
}

TEST(ArrayList, Copy)
{
	std::vector<std::vector<int>> vvec;
	std::vector<int> vec1;
	std::vector<int> vec2;

	vec1.push_back(1);
	vec1.push_back(2);

	vec2.push_back(2);
	vvec.push_back(vec1);
	vvec.push_back(vec2);

	DArrayList<int> gArrList;
	gArrList.assign(vvec);

	CArrayList<int> cArrList;
	cArrList.assign(gArrList);

	CArrayList<int> cArrList1;
	cArrList1.assign(cArrList);

	std::cout << "Device ArrayList: " << cArrList;
	std::cout << "Device ArrayList: " << cArrList1;
	std::cout << "Host ArrayList: " << gArrList;

	auto iter = cArrList[0].begin();
	EXPECT_EQ(*iter, 1);

	iter++;
	EXPECT_EQ(*iter, 2);
}

TEST(Array2D, Copy)
{
	CArray2D<int> cArr2d;
	cArr2d.resize(2, 2);
	cArr2d(0, 0) = 0;
	cArr2d(0, 1) = 1;
	cArr2d(1, 0) = 2;
	cArr2d(1, 1) = 3;

	DArray2D<int> dArr2d;
	dArr2d.assign(cArr2d);

	CArray2D<int> cArr2d2;
	cArr2d2.assign(dArr2d);

	EXPECT_EQ(cArr2d2(0, 1), 1);

	cArr2d2.assign(cArr2d);
	EXPECT_EQ(cArr2d2(0, 1), 1);
}

TEST(Array3D, Copy)
{
	CArray3D<int> cArr3d;
	cArr3d.resize(3, 3, 3);
	int ind = 0;
	for (int k = 0; k < 3; k++)
	{
		for (int j = 0; j < 3; j++)
		{
			for (int i = 0; i < 3; i++)
			{
				cArr3d(i, j, k) = ind;
				ind++;
			}
		}
	}

	DArray3D<int> dArr3d;
	dArr3d.assign(cArr3d);

	CArray3D<int> cArr3d_1,cArr3d_2;
	cArr3d_1.assign(cArr3d);
	cArr3d_2.assign(dArr3d);

	ind = 0;
	for (int k = 0; k < 3; k++)
	{
		for (int j = 0; j < 3; j++)
		{
			for (int i = 0; i < 3; i++)
			{
				EXPECT_EQ(cArr3d(i, j, k), ind);
				EXPECT_EQ(cArr3d_1(i, j, k), ind);
				EXPECT_EQ(cArr3d_2(i, j, k), ind);
				ind++;
			}
		}
	}
}

TEST(Array3D, assign)
{
	CArray3D<int> cArr3d(3, 4, 5);
	cArr3d.assign(1);
	EXPECT_EQ(cArr3d.size() == 60, true);
	EXPECT_EQ(cArr3d.nx() == 3, true);
	EXPECT_EQ(cArr3d.ny() == 4, true);
	EXPECT_EQ(cArr3d.nz() == 5, true);
	EXPECT_EQ(cArr3d(0, 0, 0) == 1, true);

	cArr3d.assign(3);
	EXPECT_EQ(cArr3d(0, 0, 0) == 3, true);

	cArr3d.assign(1, 2, 3, 2);
	EXPECT_EQ(cArr3d.size() == 6, true);
	EXPECT_EQ(cArr3d.nx() == 1, true);
	EXPECT_EQ(cArr3d.ny() == 2, true);
	EXPECT_EQ(cArr3d.nz() == 3, true);
	EXPECT_EQ(cArr3d(0, 0, 0) == 2, true);
}