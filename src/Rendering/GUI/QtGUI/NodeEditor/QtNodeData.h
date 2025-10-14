#pragma once

#include "nodes/QNodeDataModel"

#include "NodePort.h"
#include "ModulePort.h"

using dyno::NodePort;
using dyno::ModulePort;
using dyno::Node;
using dyno::Module;

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
								 PortShape::Diamond};
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
								 PortShape::Diamond};
		}

		inline std::shared_ptr<Node> getNode() { return export_node; }

		bool isEmpty() { return export_node == nullptr; }

 		bool sameType(QtNodeData& nodeData) const override;

	private:
		std::shared_ptr<Node> export_node = nullptr;
	};

	class QtImportModule : public QtNodeData
	{
	public:

		QtImportModule()
		{}

		QtImportModule(ModulePort* n)
			: module_port(n)
		{}

		NodeDataType type() const override
		{
			return NodeDataType{ "moduleport",
								 "ModulePort",
								 PortShape::Diamond };
		}

		ModulePort* getModulePort() { return module_port; }

		bool isEmpty() { return module_port == nullptr; }

		bool sameType(QtNodeData& nodeData) const override;

	private:

		ModulePort* module_port = nullptr;
	};

	class QtExportModule : public QtNodeData
	{
	public:

		QtExportModule()
		{}

		QtExportModule(std::shared_ptr<Module> n)
			: export_module(n)
		{}

		NodeDataType type() const override
		{
			return NodeDataType{ "moduleexport",
								 "ModuleExport",
								 PortShape::Diamond };
		}

		inline std::shared_ptr<Module> getModule() { return export_module; }

		bool isEmpty() { return export_module == nullptr; }

		bool sameType(QtNodeData& nodeData) const override;

	private:
		std::shared_ptr<Module> export_module = nullptr;
	};
}

