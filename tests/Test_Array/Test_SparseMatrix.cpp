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
	//Pair<int, double> ptrpair1[4];
	//Pair<int, double> ptrpair2[4];
	//double ptrdouble[4];
    //Map<int, double> ptrmap1[4];
	//Map<int, double> ptrmap2[4];

	//printf("the ptrpair1 is: %x \n", ptrpair1);
	//printf("the ptrpair1 is: %x \n\n", &(ptrpair1[1]));
	//printf("the ptrpair2 is: %x \n", ptrpair2);
	//printf("the ptrpair2 is: %x \n\n", &(ptrpair2[1]));
	//printf("the ptrdouble is: %x \n", ptrdouble);
	//printf("the ptrdouble is: %x \n\n", &(ptrdouble[1]));
	//printf("the ptrmap1 is: %x \n", ptrmap1);
	//printf("the ptrmap1 is: %x \n\n", &(ptrmap1[1]));
	//printf("the ptrmap2 is: %x \n", ptrmap2);
	//printf("the ptrmap2 is: %x \n\n", &(ptrmap2[1]));


	std::vector<std::map<int, float>> vecmap;
	for (int i = 0; i < 4; i++)
	{
		std::map<int, float> mymap;
		if (i == 0)
		{
			mymap.insert(std::pair<int, float>(0, 1));
			mymap.insert(std::pair<int, float>(3, 5));
		}
		else if (i == 2)
		{
			mymap.insert(std::pair<int, float>(2, 2));
			mymap.insert(std::pair<int, float>(3, 6));
		}
		else if (i == 3)
		{
			mymap.insert(std::pair<int, float>(2, 2));
			mymap.insert(std::pair<int, float>(3, 1));
		}
		vecmap.push_back(mymap);
		//std::cout << std::endl << std::endl;
	}

	DArrayMap<float> A;
	A.assign(vecmap);
	std::cout << "A ArrayMap: " << A << std::endl;

	CArray<float> Cb(4);
	Cb.reset();
	Cb[0] = -14.0;
	Cb[1] = 0.0;
	Cb[2] = -15.0;
	Cb[3] = 0.0;

	DArray<float> b;
	b.assign(Cb);

	std::cout << "b: "<< std::endl <<Cb << std::endl;

	SparseMatrix<float> solver(A, b);
	std::cout << "it is ok here !" << std::endl;
	solver.CGLS(1000, 0.000001);

	CArray <float> Cx;
	Cx.assign(solver.X());

	std::cout << "x: " << std::endl << Cx << std::endl;
	solver.clear();
}






