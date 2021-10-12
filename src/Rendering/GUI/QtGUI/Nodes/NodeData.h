#pragma once

#include "QtBlockDataModel.h"
#include "NodePort.h"

using QtNodes::BlockDataType;
using QtNodes::BlockData;

using dyno::NodePort;
using dyno::Node;

/// The class can potentially incapsulate any user data which
/// need to be transferred within the Node Editor graph
class NodeImportData : public BlockData
{
public:

	NodeImportData()
	{}

	NodeImportData(NodePort* n)
		: node_port(n)
	{}

	BlockDataType type() const override
	{
		return BlockDataType{ "nodeport",
							 "NodePort" };
	}

	NodePort* getNodePort() { return node_port; }

	bool isEmpty() { return node_port == nullptr; }

	bool isKindOf(BlockData &nodedata) const override;

private:

	NodePort* node_port = nullptr;
};


class NodeExportData : public BlockData
{
public:

	NodeExportData()
	{}

	NodeExportData(std::shared_ptr<Node> n)
		: export_node(n)
	{}

	BlockDataType type() const override
	{
		return BlockDataType{ "nodeexport",
							 "NodeExport" };
	}

	inline std::shared_ptr<Node> getNode() { return export_node; }

	bool isEmpty() { return export_node == nullptr; }

	bool isKindOf(BlockData &nodedata) const override;

private:

	std::shared_ptr<Node> export_node = nullptr;
};
