#include "gtest/gtest.h"
#include "STL/Set.h"
#include "STL/MultiSet.h"
#include "STL/MultiMap.h"
#include "STL/Stack.h"
#include "STL/List.h"

using namespace dyno;

TEST(Set, insert)
{
	int buf[20];
	Set<int> set;
	set.reserve(buf, 20);

	set.insert(10);
	set.insert(20);
	set.insert(20);

	EXPECT_EQ(set.count(20), 1);

	Stack<int> stack;
	stack.reserve(buf, 20);

	stack.push(10);
	stack.push(11);
	stack.push(11);
	stack.push(11);
	stack.push(12);
	stack.pop();

	EXPECT_EQ(stack.top(), 11);
	EXPECT_EQ(stack.count(11), 3);

	List<int> list;
	list.reserve(buf, 10);
	list.insert(10);//head->10
	list.insert(9);

	EXPECT_EQ(list.front(), 10);
	EXPECT_EQ(list.back(), 9);
	EXPECT_EQ(list.size(), 2);

	MultiSet<int> multSet;
	multSet.reserve(buf, 20);

	multSet.insert(10);
	multSet.insert(15);
	multSet.insert(15);

	EXPECT_EQ(multSet.count(15), 2);

	MultiSet<int> multSet_1;
	EXPECT_EQ(multSet_1.count(1), 0);

	Pair<int, int> pairs[20];
	MultiMap<int, int> multiMap;
	multiMap.reserve(pairs, 20);

	multiMap.insert(Pair<int, int>(10, 101));
	multiMap.insert(Pair<int, int>(20, 5));
	multiMap.insert(Pair<int, int>(5, 3));

	EXPECT_EQ(multiMap[10], 101);
	EXPECT_EQ(multiMap.size(), 3);
}