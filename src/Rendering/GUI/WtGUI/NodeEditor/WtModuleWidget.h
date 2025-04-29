#pragma once

#include "json.hpp"
#include "WtNodeDataModel.h"
#include "WtFieldData.h"

#include "Module.h"

#include <iostream>

using dyno::Module;
using dyno::FBase;

class WtModuleWidget :public WtNodeDataModel
{
public:
	WtModuleWidget(std::shared_ptr<Module> base = nullptr);

	virtual ~WtModuleWidget() {}

public:

	std::string caption() const override;

	bool captionVisible() const override;

	std::string name() const override;

	void setName(std::string name) { m_name = name; }

	std::string portCaption(PortType portType, PortIndex portIndex) const override;

	std::string nodeTips() const override;

	std::string portTips(PortType portType, PortIndex portIndex) const override;

	std::string validationMessage() const override;

	unsigned int nPorts(PortType portType) const override;

	bool hotkeyEnabled() const override { return false; }

	bool allowExported() const override { return false; }

	bool portCaptionVisible(PortType portType, PortIndex portIndex) const override;

	std::shared_ptr<WtNodeData> outData(PortIndex port) override;

	void setInData(std::shared_ptr<WtNodeData> data, PortIndex portIndex) override;

	bool tryInData(PortIndex portIndex, std::shared_ptr<WtNodeData> nodeData) override;

	NodeDataType dataType(PortType portType, PortIndex portIndex) const override;

	//QWidget* embeddedWidget() override { return nullptr; }

	NodeValidationState validationState() const override;

	std::shared_ptr<Module> getModule();

	void enableEditing();

	void disableEditing();

protected:
	virtual void updateModule();

protected:

	using OutFieldPtr = std::vector<std::shared_ptr<WtFieldData>>;

	using InFieldPtr = std::vector<std::shared_ptr<WtFieldData>>;

	InFieldPtr input_fields;

	OutFieldPtr output_fields;

	std::string m_name;

	std::shared_ptr<Module> m_Module = nullptr;

	NodeValidationState modelValidationState = NodeValidationState::Warning;

	std::string modelValidationError = "Missing or incorrect inputs";

private:

	FBase* getField(PortType portType, PortIndex portIndex) const;

	std::vector<FBase*>& getOutputFields();

	std::vector<FBase*>& getInputFields();

private:

	bool mEditingEnabled = true;
};