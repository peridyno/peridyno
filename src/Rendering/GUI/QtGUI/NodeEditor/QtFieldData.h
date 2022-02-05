#pragma once

#include "nodes/QNodeDataModel"
#include "FBase.h"

using dyno::FBase;

namespace Qt
{

	/// The class can potentially incapsulate any user data which
	/// need to be transferred within the Node Editor graph
	class QtFieldData : public QtNodeData
	{
	public:

		QtFieldData()
		{}

		QtFieldData(FBase* f)
			: field(f)
		{}

		NodeDataType type() const override
		{
			return NodeDataType{ "field",
								 "Field" };
		}

		FBase* getField() { return field; }

		bool isEmpty() { return field == nullptr; }

		bool isKindOf(QtNodeData& nodedata) const
		{
			return true;
		}

	private:

		FBase* field = nullptr;
	};
}

