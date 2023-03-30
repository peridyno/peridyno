#include "gtest/gtest.h"
//#include "STL/Map.h"
//#include "STL/Pair.h"
//#include "STL/List.h"
#include "Array/ArrayMap.h"
#include <vector>
#include <map>

using namespace dyno;

TEST(ArrayMap, CPU_GPU)
{
	std::map<int, double> mymap;
	mymap.insert(std::pair<int, double>(1, 1.1));
	mymap.insert(std::pair<int, double>(4, 4.4));
	mymap.insert(std::pair<int, double>(2, 2.2));
	mymap.insert(std::pair<int, double>(2, 2.3));
	std::map<int, double>::iterator map_it = mymap.begin();
	for (int i = 0; i < mymap.size(); i++)
	{
		std::cout << "the mymap[" << i << "] is: " << map_it->first << " " << map_it->second << std::endl;
		map_it++;
	}

	//测试map中“=”的使用
	std::map<int, double> mymap0=mymap;
	std::map<int, double>::iterator map_it0 = mymap0.begin();
	for (int i = 0; i < mymap0.size(); i++)
	{
		std::cout << "the mymap[" << i << "] is: " << map_it0->first << " " << map_it0->second << std::endl;
		map_it0++;
	}


	std::map<int, double> mymap1;
	mymap1.insert(std::pair<int, double>(1, 1.1));
	mymap1.insert(std::pair<int, double>(4, 4.4));
	mymap1.insert(std::pair<int, double>(2, 2.2));
	mymap1.insert(std::pair<int, double>(3, 3.3));
	std::map<int, double>::iterator map_it1 = mymap1.begin();
	for (int i = 0; i < mymap1.size(); i++)
	{
		std::cout << "the mymap[" << i << "] is: " << map_it1->first << " " << map_it1->second << std::endl;
		map_it1++;
	}

	//空map
	std::map<int, double> mymap2;

	std::vector<std::map<int, double>> vecmap;
	vecmap.push_back(mymap);
	vecmap.push_back(mymap2);
	vecmap.push_back(mymap1);

	DArrayMap<double> dArrMap;
	dArrMap.assign(vecmap);

	CArrayMap<double> cArrMap;
	cArrMap.assign(dArrMap);

	std::cout << "Host ArrayMap: " << cArrMap;
	std::cout << "Device ArrayMap: " << dArrMap;


	EXPECT_EQ(cArrMap[0].size(), 3);
	EXPECT_EQ(cArrMap[1].size(), 0);
	EXPECT_EQ(cArrMap[2].size(), 4);
}





