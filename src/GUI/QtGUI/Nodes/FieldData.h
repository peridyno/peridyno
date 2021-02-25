#pragma once

#include "QtBlockDataModel.h"
#include "Framework/Field.h"

using QtNodes::BlockDataType;
using QtNodes::BlockData;

using dyno::Field;

/// The class can potentially incapsulate any user data which
/// need to be transferred within the Node Editor graph
class FieldData : public BlockData
{
public:

	FieldData()
	{}

	FieldData(Field* f)
		: field(f)
	{}

	BlockDataType type() const override
	{
		return BlockDataType{ "field",
							 "Field" };
	}

	Field* getField() { return field; }

	bool isEmpty() { return field == nullptr; }

	bool isKindOf(BlockData &nodedata) const override
	{
		return true;
	}

private:

	Field* field = nullptr;
};
