#pragma once

#include <QtCore/QObject>
#include <QtCore/QJsonObject>
#include <QtWidgets/QLabel>

#include "QtBlockDataModel.h"
#include "Module.h"

#include <iostream>

class FieldData;

using dyno::Module;
using dyno::FBase;

using QtNodes::PortType;
using QtNodes::PortIndex;
using QtNodes::BlockData;
using QtNodes::BlockDataType;
using QtNodes::QtBlockDataModel;
using QtNodes::ValidationState;

namespace QtNodes
{

	/// The model dictates the number of inputs and outputs for the Node.
	/// In this example it has no logic.
	class QtModuleWidget : public QtBlockDataModel
	{
		Q_OBJECT

	public:
		QtModuleWidget(Module* base = nullptr);

		virtual	~QtModuleWidget() {}

	public:

		QString caption() const override;

		QString name() const override;
		void setName(QString name) { m_name = name; }

		QString	portCaption(PortType portType, PortIndex portIndex) const override;

		QString	validationMessage() const override;


		unsigned int nPorts(PortType portType) const override;


		bool portCaptionVisible(PortType portType, PortIndex portIndex) const override;

		std::shared_ptr<BlockData> outData(PortIndex port) override;
		std::shared_ptr<BlockData> inData(PortIndex port) override;

		void setInData(std::shared_ptr<BlockData> data, PortIndex portIndex) override;

		BlockDataType dataType(PortType portType, PortIndex portIndex) const override;


		QWidget* embeddedWidget() override { return nullptr; }

		ValidationState validationState() const override;

		Module* getModule();

	protected:
		virtual void updateModule();

	protected:

		using OutFieldPtr = std::vector<std::shared_ptr<FieldData>>;
		using InFieldPtr = std::vector<std::weak_ptr<FieldData>>;

		InFieldPtr input_fields;
		OutFieldPtr output_fields;

		QString m_name;

		Module* m_module = nullptr;

		ValidationState modelValidationState = ValidationState::Warning;
		QString modelValidationError = QString("Missing or incorrect inputs");

	private:

		FBase* getField(PortType portType, PortIndex portIndex) const;

		std::vector<FBase*>& getOutputFields();
		std::vector<FBase*>& getInputFields();
	};
}
