#include "WtNodeData.hpp"

bool WtImportNode::sameType(WtNodeData& nodeData) const
{
	try
	{
		auto& out_data = dynamic_cast<WtExportNode&>(nodeData);
	}
	catch (std::bad_cast)
	{
		return false;
	}

	auto& out_data = dynamic_cast<WtExportNode&>(nodeData);

	return node_port->isKindOf(out_data.getNode().get());
}

bool WtExportNode::sameType(WtNodeData& nodeData) const
{
	try
	{
		auto& in_data = dynamic_cast<WtImportNode&>(nodeData);
	}
	catch (std::bad_cast)
	{
		return false;
	}

	auto& in_data = dynamic_cast<WtImportNode&>(nodeData);

	return in_data.getNodePort()->isKindOf(export_node.get());
}