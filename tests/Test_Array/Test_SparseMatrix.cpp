#include "gtest/gtest.h"
//#include "STL/Map.h"
//#include "STL/Pair.h"
#include "Array/ArrayMap.h"
#include "Matrix/SparseMatrix.h"
#include <vector>

using namespace dyno;

TEST(SparseMatrix, CPU)
{
	//Pair<int, double> pair11(1, 1), pair12(2, 2), pair14(4, 5), pair32(2, 3), pair33(3, 2), pair34(4, 6), pair43(3, 2), pair44(4, 1);
	//std::vector<std::vector<Pair<int, double>>> vecmaps;

	for (int i = 0; i < 4; i++)
	{
		std::cout << "i is: " << i << std::endl;
		std::vector<Pair<int, double>> vecmap;
		if (i == 0)
		{
			Map<int, double> mymap1;
			Pair<int, double>* pairptr1 = (Pair<int, double>*)malloc(sizeof(Pair<int, double>) * 4);
			mymap1.reserve(pairptr1, 4);
			mymap1.insert(pair11);
			mymap1.insert(pair12);
			mymap1.insert(pair14);

			for (int j = 0; j < mymap1.size(); j++)
			{
				vecmap.push_back(mymap1[j]);
			}
			free(pairptr1);
		}
		else if (i == 2)
		{
			Map<int, double> mymap1;
			Pair<int, double>* pairptr1 = (Pair<int, double>*)malloc(sizeof(Pair<int, double>) * 4);
			mymap1.reserve(pairptr1, 4);
			mymap1.insert(pair32);
			mymap1.insert(pair33);
			mymap1.insert(pair34);

			for (int j = 0; j < mymap1.size(); j++)
			{
				vecmap.push_back(mymap1[j]);
			}
			free(pairptr1);
		}
		else if (i == 3)
		{
			Map<int, double> mymap1;
			Pair<int, double>* pairptr1 = (Pair<int, double>*)malloc(sizeof(Pair<int, double>) * 4);
			mymap1.reserve(pairptr1, 4);
			mymap1.insert(pair43);
			mymap1.insert(pair44);

			for (int j = 0; j < mymap1.size(); j++)
			{
				vecmap.push_back(mymap1[j]);
			}
			free(pairptr1);
		}
		for (int j = 0; j < vecmap.size(); j++)
		{
			std::cout << vecmap[j].first << ": " << vecmap[j].second << std::endl;
		}
		vecmaps.push_back(vecmap);
		std::cout << std::endl << std::endl;
	}

	//DArrayMap<double> A;
	//A.assign(vecmaps);
	//std::cout << "A ArrayMap: " << A << std::endl;

	//CArray<double> Cb(4);
	//Cb.reset();
	//Cb[0] = 1.0;
	//Cb[1] = 0.0;
	//Cb[2] = 1.0;
	//Cb[3] = 1.0;

	//DArray<double> b;
	//b.assign(Cb);

	//std::cout << "b: "<< std::endl <<Cb << std::endl;

	//SparseMatrix<double> solver;
	//solver.Initialize(A, b);
	//std::cout << "it is ok here !" << std::endl;
	//solver.CGLS(100, 0.001);

	//CArray<double> Cx;
	//Cx.assign(solver.X());

	//std::cout << "x: " << std::endl << Cx << std::endl;
}




