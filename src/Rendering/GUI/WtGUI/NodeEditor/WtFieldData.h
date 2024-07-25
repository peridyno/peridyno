#pragma once

#include "FBase.h"
#include "WtNodeDataModel.h"

using dyno::FBase;

class WtFieldData : public WtNodeData
{
public:
	WtFieldData() {}
	WtFieldData(FBase* f)
		: field(f) {}

	NodeDataType type() const override
	{
		return NodeDataType{ "field", "Field" };
	}

	FBase* getField() { return field; }

	bool isEmpty() { return field == nullptr; }

	bool isKindOf(WtNodeData& nodedata) const
	{
		return true;
	}

private:
	FBase* field = nullptr;
};