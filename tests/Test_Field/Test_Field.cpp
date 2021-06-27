#include "gtest/gtest.h"

#include "CalculateArea.h"

using namespace dyno;

TEST(ModuleField, connect)
{
	CalculateArea calArea("m1");

	calArea.inWidth()->setValue(12.0f);
	EXPECT_EQ(calArea.outArea()->isEmpty(), true);

	calArea.inHeight()->setValue(5.0f);
	calArea.update();
	EXPECT_EQ(calArea.outArea()->getData(), float(60));

	VarField<float> inputWidth;
	inputWidth.setValue(30.0f);
	inputWidth.connect(calArea.inWidth());

	calArea.update();
	EXPECT_EQ(calArea.outArea()->getData(), float(150));

	CalculateArea calArea1("m1");

	inputWidth.connect(calArea1.inWidth());
	EXPECT_EQ(inputWidth.sinkSize(), 2);

	VarField<float> inputWidth1;
	inputWidth1.setValue(10.0f);
	inputWidth1.connect(calArea1.inWidth());
	EXPECT_EQ(inputWidth.sinkSize(), 1);
	EXPECT_EQ(inputWidth1.sinkSize(), 1);

// 	ArrayField<float, DeviceType::GPU> arrField;
// 	DArray<float> dArr;
// 	dArr.resize(20);
// 
// 	std::shared_ptr<float> A = nullptr;
// 	std::shared_ptr<float>& B = A;
// 
// 	EXPECT_EQ(arrField.isEmpty(), true);
}
