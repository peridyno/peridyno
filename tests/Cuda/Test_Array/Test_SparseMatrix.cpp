#include "gtest/gtest.h"
//#include "STL/Map.h"
//#include "STL/Pair.h"
#include "Array/ArrayMap.h"
#include "Matrix/SparseMatrix.h"
#include <vector>
#include <map>

using namespace dyno;

TEST(SparseMatrix, CPU)
{
	std::vector<std::map<int, float>> vecmap,vecmap1;
	for (int i = 0; i < 5; i++)
	{
		std::map<int, float> mymap;
		if (i == 0)
		{
			mymap.insert(std::pair<int, float>(0, 1.2));
			mymap.insert(std::pair<int, float>(1, 2));
			mymap.insert(std::pair<int, float>(3, 1.67));
			mymap.insert(std::pair<int, float>(4, 3.1));
		}
		else if (i == 1)
		{
			mymap.insert(std::pair<int, float>(0, 7.2));
			mymap.insert(std::pair<int, float>(1, 0.3));
			mymap.insert(std::pair<int, float>(3, 0.5));
			mymap.insert(std::pair<int, float>(4, 5.7));
		}
		else if (i == 3)
		{
			mymap.insert(std::pair<int, float>(0, 2.1));
			mymap.insert(std::pair<int, float>(1, 3.2));
			mymap.insert(std::pair<int, float>(3, 7.3));
			mymap.insert(std::pair<int, float>(4, 5.22));
		}
		else if (i == 4)
		{
			mymap.insert(std::pair<int, float>(0, 3.71));
			mymap.insert(std::pair<int, float>(1, 4.47));
			mymap.insert(std::pair<int, float>(3, 3.76));
			mymap.insert(std::pair<int, float>(4, 5.29));
		}

		vecmap.push_back(mymap);
		//std::cout << std::endl << std::endl;
	}

	vecmap1.resize(vecmap.size());
	for (int i = 0; i < vecmap.size(); i++)
	{
		for (auto map_it = vecmap[i].begin(); map_it != vecmap[i].end(); map_it++)
			vecmap1[map_it->first].insert({i, map_it->second });
	}


	DArrayMap<float> A,AT;
	A.assign(vecmap);
	AT.assign(vecmap1);
	std::cout << "A ArrayMap: " << A << std::endl;
	std::cout << "AT ArrayMap: " << AT << std::endl;

	CArray<float> Cb(5);
	Cb.reset();
	Cb[0] =1.6;
	Cb[1] = 3.26;
	Cb[2] = 0;
	Cb[3] = 9.1;
	Cb[4] = 6.67;

	//DArray<float> b;
	//b.assign(Cb);

	//std::cout << "b: "<< std::endl <<Cb << std::endl;

	//SparseMatrix<float> solver(A, b);
	//std::cout << "it is ok here !" << std::endl;
	//solver.CGLS(1000, 0.000001);

	//CArray <float> Cx;
	//Cx.assign(solver.X());

	//std::cout << "x: " << std::endl << Cx << std::endl;
	//solver.clear();
}






