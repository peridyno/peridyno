#pragma once

#include "Node.h"
#include "WtNodeData.hpp"
#include "WtNodeDataModel.h"
#include "WtFieldData.h"

using dyno::Node;
using dyno::FBase;

class WtNodeWidget : public WtNodeDataModel
{
public:
	WtNodeWidget(std::shared_ptr<Node> base = nullptr);
	virtual ~WtNodeWidget();

public:
	std::string caption() const override;

	std::string name() const override;

	std::string	nodeTips() const override;

	std::string	portCaption(PortType portType, PortIndex portIndex) const override;

	std::string	portTips(PortType portType, PortIndex portIndex) const override;

	std::string	validationMessage() const override;

	unsigned int nPorts(PortType portType) const override;

	bool portCaptionVisible(PortType portType, PortIndex portIndex) const override;

	bool allowExported() const override { return true; }

	/**
 * @brief To test whether nodaData can be set as the input data for portIndex
 *
 * @param portIndex Input index
 * @param nodeData Input data
 * @return true
 * @return false
 */
	bool tryInData(PortIndex portIndex, std::shared_ptr<WtNodeData> nodeData) override;

	void setInData(std::shared_ptr<WtNodeData> data, PortIndex portIndex) override;

	NodeDataType dataType(PortType portType, PortIndex portIndex) const override;

	//QWidget* embeddedWidget() override { return nullptr; }

	NodeValidationState validationState() const override;

	WtNodeDataModel::ConnectionPolicy portInConnectionPolicy(PortIndex portIndex) const override;

	std::shared_ptr<Node> getNode();

	std::shared_ptr<WtNodeData> outData(PortIndex port) override;

	std::vector<FBase*>& getOutputFields() const;
	std::vector<FBase*>& getInputFields() const;
	/**
	* @brief When enabled, the scenegraph can be updated as long as the corresponding GUI is updated.
	*/
	void enableEditing();

	/**
	 * @brief When disabled, the scenegraph can not be affected by the corresponding GUI.
	 */
	void disableEditing();

protected:
	virtual void updateModule();

protected:
	using ExportNodePtr = std::shared_ptr<WtExportNode>;
	using ImportNodePtr = std::vector<std::shared_ptr<WtImportNode>>;

	ImportNodePtr mNodeInport;
	ExportNodePtr mNodeExport;

	using OutFieldPtr = std::vector<std::shared_ptr<WtFieldData>>;
	using InFieldPtr = std::vector<std::shared_ptr<WtFieldData>>;

	InFieldPtr mFieldInport;
	OutFieldPtr mFieldExport;

	std::shared_ptr<Node> mNode = nullptr;

	NodeValidationState modelValidationState = NodeValidationState::Valid;
	std::string modelValidationError = std::string("Missing or incorrect inputs");

private:
	bool mEditingEnabled = true;
};