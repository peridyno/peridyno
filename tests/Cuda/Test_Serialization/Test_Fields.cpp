#include "gtest/gtest.h"

#include "Field.h"
#include "FilePath.h"
using namespace dyno;

TEST(Fields, serialize)
{
	FVar<FilePath> varPath;
	varPath.setValue(FilePath("Test"));
	EXPECT_EQ(varPath.serialize(), std::string("Test"));

	FVar<bool> varBool;
	varBool.setValue(true);
	EXPECT_EQ(varBool.serialize(), std::string("true"));

	FVar<int> varInt;
	varInt.setValue(-1);
	EXPECT_EQ(varInt.serialize(), std::string("-1"));

	FVar<uint> varUInt;
	varUInt.setValue(1);
	EXPECT_EQ(varUInt.serialize(), std::string("1"));
}

TEST(Fields, deserialize)
{
	FVar<FilePath> varPath;
	varPath.deserialize("Test");	
	EXPECT_EQ(varPath.getValue(), std::string("Test"));

	FVar<bool> varBool;
	varBool.deserialize("true");
	EXPECT_EQ(varBool.getValue(), true);

	FVar<int> varInt;
	varInt.deserialize("-1");
	EXPECT_EQ(varInt.getValue(), -1);

	FVar<uint> varUInt;
	varUInt.deserialize("1");
	EXPECT_EQ(varUInt.getValue(), 1);

	FVar<Vec3i> varVec3i;
	varVec3i.deserialize("1 2 3");
	Vec3i val = varVec3i.getValue();
	EXPECT_EQ(val.x, 1);
	EXPECT_EQ(val.y, 2);
	EXPECT_EQ(val.z, 3);
}