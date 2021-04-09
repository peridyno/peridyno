#include "gtest/gtest.h"
#include "ModuleFields.h"

using namespace dyno;

TEST(FieldBase, connect)
{
	ModuleFields m1("m1");
	ModuleFields m2("m2");

	m1.varArea()->connect(m2.varArea());

	m1.varWidth()->setValue(12.0f);
	m1.varHeight()->setValue(5.0f);

	EXPECT_EQ(m1.varArea()->getData(), float(60));

	ArrayField<float, DeviceType::GPU> arrField;
	DArray<float> dArr;
	dArr.resize(20);

	std::shared_ptr<float> A = nullptr;
	std::shared_ptr<float>& B = A;

	EXPECT_EQ(arrField.isEmpty(), true);
}
