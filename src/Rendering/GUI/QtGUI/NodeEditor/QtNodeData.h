#pragma once

#include "nodes/QNodeDataModel"
#include "NodePort.h"

using dyno::NodePort;
using dyno::Node;

namespace Qt
{
	/// The class can potentially incapsulate any user data which
/// need to be transferred within the Node Editor graph
	class QtImportNode : public QtNodeData
	{
	public:

		QtImportNode()
		{}

		QtImportNode(NodePort* n)
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
 
 		bool sameType(QtNodeData& nodeData) const override;

	private:

		NodePort* node_port = nullptr;
	};


	class QtExportNode : public QtNodeData
	{
	public:

		QtExportNode()
		{}

		QtExportNode(std::shared_ptr<Node> n)
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

 		bool sameType(QtNodeData& nodeData) const override;

	private:
		std::shared_ptr<Node> export_node = nullptr;
	};
}

