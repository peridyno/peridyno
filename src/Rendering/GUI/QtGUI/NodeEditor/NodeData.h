#pragma once

#include "nodes/NodeDataModel"
#include "NodePort.h"

using dyno::NodePort;
using dyno::Node;

namespace Qt
{
	/// The class can potentially incapsulate any user data which
/// need to be transferred within the Node Editor graph
	class NodeImportData : public QtNodeData
	{
	public:

		NodeImportData()
		{}

		NodeImportData(NodePort* n)
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


	class NodeExportData : public QtNodeData
	{
	public:

		NodeExportData()
		{}

		NodeExportData(std::shared_ptr<Node> n)
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

		bool isToDisconnected() {
			return m_isToDisconnected;
		}

		void setDisconnected(bool connected) {
			m_isToDisconnected = connected;
		}

	private:
		bool m_isToDisconnected = false;

		std::shared_ptr<Node> export_node = nullptr;
	};
}

