#pragma once

#include "nodes/NodeDataModel"
#include "NodePort.h"

using dyno::NodePort;
using dyno::Node;

namespace Qt
{
	/// The class can potentially incapsulate any user data which
/// need to be transferred within the Node Editor graph
	class QtNodeImportData : public QtNodeData
	{
	public:

		QtNodeImportData()
		{}

		QtNodeImportData(NodePort* n)
			: node_port(n)
		{}

		NodeDataType type() const override
		{
			return NodeDataType{ "nodeport",
								 "NodePort",
								 PortShape::Bullet};
		}

		NodePort* getNodePort() { return node_port; }

		bool isEmpty() { return node_port == nullptr; }

		bool isKindOf(QtNodeData& nodedata) const;

	private:

		NodePort* node_port = nullptr;
	};


	class QtNodeExportData : public QtNodeData
	{
	public:

		QtNodeExportData()
		{}

		QtNodeExportData(std::shared_ptr<Node> n)
			: export_node(n)
		{}

		NodeDataType type() const override
		{
			return NodeDataType{ "nodeexport",
								 "NodeExport",
								 PortShape::Bullet};
		}

		inline std::shared_ptr<Node> getNode() { return export_node; }

		bool isEmpty() { return export_node == nullptr; }

		bool isKindOf(QtNodeData& nodedata) const;

	private:
		std::shared_ptr<Node> export_node = nullptr;
	};
}

