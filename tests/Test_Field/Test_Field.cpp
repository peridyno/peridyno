#include "gtest/gtest.h"

#include "CalculateArea.h"
#include "FInstance.h"

using namespace dyno;

TEST(ModuleField, connect)
{
	CalculateArea calArea("m1");

	calArea.inWidth()->setValue(12.0f);
	EXPECT_EQ(calArea.outArea()->isEmpty(), true);

	calArea.inHeight()->setValue(5.0f);
	calArea.update();
	EXPECT_EQ(calArea.outArea()->getData(), float(60));

	FVar<float> inputWidth;
	inputWidth.setValue(30.0f);
	inputWidth.connect(calArea.inWidth());

	calArea.update();
	EXPECT_EQ(calArea.outArea()->getData(), float(150));

	CalculateArea calArea1("m1");

	inputWidth.connect(calArea1.inWidth());
	EXPECT_EQ(inputWidth.sizeOfSinks(), 2);

	FVar<float> inputWidth1;
	inputWidth1.setValue(10.0f);
	inputWidth1.connect(calArea1.inWidth());
	EXPECT_EQ(inputWidth.sizeOfSinks(), 1);
	EXPECT_EQ(inputWidth1.sizeOfSinks(), 1);

// 	ArrayField<float, DeviceType::GPU> arrField;
// 	DArray<float> dArr;
// 	dArr.resize(20);
// 
// 	std::shared_ptr<float> A = nullptr;
// 	std::shared_ptr<float>& B = A;
// 
// 	EXPECT_EQ(arrField.isEmpty(), true);
}


class OA : public Object {
public:
	OA() {};

	int val = 1;
};

class OB : public OA
{
public:
	OB() { val = 2; };

private:

};

TEST(FInstance, connect)
{
	FInstance<OA> oA;
	oA.setDataPtr(std::make_shared<OA>());
	FInstance<OB> oB;
	oB.setDataPtr(std::make_shared<OB>());
	oB.connect(&oA);

	FInstance<OB> oA2;
	oB.connect(&oA2);

	auto oBData = oB.getData();
	auto oAData = oA.getData();

	EXPECT_EQ(oA.getData().val == 2, true);
	EXPECT_EQ(oA.isEmpty(), false);
	EXPECT_EQ(oB.sizeOfSinks() == 2, true);

	FInstance<OA> oA1;
	oA1.setDataPtr(std::make_shared<OA>());
	FInstance<OB> oB1;
	//oB1.allocate();

	oA1.connect(&oB1);
	EXPECT_EQ(oB1.isEmpty(), true);
}