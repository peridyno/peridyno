#pragma once

#include "QtBlockDataModel.h"
#include "FBase.h"

using QtNodes::BlockDataType;
using QtNodes::BlockData;

using dyno::FBase;

/// The class can potentially incapsulate any user data which
/// need to be transferred within the Node Editor graph
class FieldData : public BlockData
{
public:

	FieldData()
	{}

	FieldData(FBase* f)
		: field(f)
	{}

	BlockDataType type() const override
	{
		return BlockDataType{ "field",
							 "Field" };
	}

	FBase* getField() { return field; }

	bool isEmpty() { return field == nullptr; }

	bool isKindOf(BlockData &nodedata) const override
	{
		return true;
	}

private:

	FBase* field = nullptr;
};
