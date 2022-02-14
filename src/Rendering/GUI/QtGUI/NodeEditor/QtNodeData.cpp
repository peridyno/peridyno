#include "QtNodeData.h"

namespace Qt
{
	bool QtImportNode::sameType(QtNodeData& nodeData) const
	{
		try
		{
			auto& out_data = dynamic_cast<QtExportNode&>(nodeData);
		}
		catch (std::bad_cast)
		{
			return false;
		}

		auto& out_data = dynamic_cast<QtExportNode&>(nodeData);

		return node_port->isKindOf(out_data.getNode().get());
	}

	bool QtExportNode::sameType(QtNodeData& nodeData) const
	{
		try
		{
			auto& in_data = dynamic_cast<QtImportNode&>(nodeData);
		}
		catch (std::bad_cast)
		{
			return false;
		}

		auto& in_data = dynamic_cast<QtImportNode&>(nodeData);

		return in_data.getNodePort()->isKindOf(export_node.get());
	}
}

