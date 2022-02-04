#include "QtNodeData.h"

namespace Qt
{
	bool QtNodeImportData::isKindOf(QtNodeData& nodedata) const
	{
		try
		{
			auto& out_data = dynamic_cast<QtNodeExportData&>(nodedata);
		}
		catch (std::bad_cast)
		{
			return false;
		}

		auto& out_data = dynamic_cast<QtNodeExportData&>(nodedata);
		return node_port->isKindOf(out_data.getNode());
	}

	bool QtNodeExportData::isKindOf(QtNodeData& nodedata) const
	{
		try
		{
			auto& in_data = dynamic_cast<QtNodeImportData&>(nodedata);
		}
		catch (std::bad_cast)
		{
			return false;
		}

		auto& in_data = dynamic_cast<QtNodeImportData&>(nodedata);
		return in_data.getNodePort()->isKindOf(export_node);
	}
}

