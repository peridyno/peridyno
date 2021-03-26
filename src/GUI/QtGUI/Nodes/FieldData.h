#pragma once

#include "QtBlockDataModel.h"
#include "Framework/FieldBase.h"

using QtNodes::BlockDataType;
using QtNodes::BlockData;

using dyno::FieldBase;

/// The class can potentially incapsulate any user data which
/// need to be transferred within the Node Editor graph
class FieldData : public BlockData
{
public:

	FieldData()
	{}

	FieldData(FieldBase* f)
		: field(f)
	{}

	BlockDataType type() const override
	{
		return BlockDataType{ "field",
							 "Field" };
	}

	FieldBase* getField() { return field; }

	bool isEmpty() { return field == nullptr; }

	bool isKindOf(BlockData &nodedata) const override
	{
		return true;
	}

private:

	FieldBase* field = nullptr;
};
