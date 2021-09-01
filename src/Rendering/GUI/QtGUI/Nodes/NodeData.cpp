#include "NodeData.h"

bool NodeImportData::isKindOf(BlockData &nodedata) const
{
	try
	{
		auto& out_data = dynamic_cast<NodeExportData&>(nodedata);
	}
	catch (std::bad_cast)
	{
		return false;
	}

	auto& out_data = dynamic_cast<NodeExportData&>(nodedata);
	return node_port->isKindOf(out_data.getNode());
}

bool NodeExportData::isKindOf(BlockData &nodedata) const
{
	try
	{
		auto& in_data = dynamic_cast<NodeImportData&>(nodedata);
	}
	catch (std::bad_cast)
	{
		return false;
	}

	auto& in_data = dynamic_cast<NodeImportData&>(nodedata);
	return in_data.getNodePort()->isKindOf(export_node);
}
