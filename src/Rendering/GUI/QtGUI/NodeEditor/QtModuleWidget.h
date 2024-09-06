#pragma once

#include <QtCore/QObject>
#include <QtCore/QJsonObject>
#include <QtWidgets/QLabel>

#include "nodes/QNodeDataModel"

#include "QtFieldData.h"
#include "Module.h"

#include <iostream>

using dyno::Module;
using dyno::FBase;

namespace Qt
{

	/// The model dictates the number of inputs and outputs for the Node.
	/// In this example it has no logic.
	class QtModuleWidget : public QtNodeDataModel
	{
		Q_OBJECT

	public:
		QtModuleWidget(std::shared_ptr<Module> base = nullptr);

		virtual	~QtModuleWidget() {}

	public:

		QString caption() const override;

		/// whether to hide caption in GUI
		bool captionVisible() const override;

		QString name() const override;
		void setName(QString name) { m_name = name; }

		QString	portCaption(PortType portType, PortIndex portIndex) const override;

		QString	nodeTips() const override;

		QString	portTips(PortType portType, PortIndex portIndex) const override;

		QString	validationMessage() const override;


		unsigned int nPorts(PortType portType) const override;

		bool hotkeyEnabled() const override { return false; }

		bool allowExported() const override { return false; }

		bool portCaptionVisible(PortType portType, PortIndex portIndex) const override;

		std::shared_ptr<QtNodeData> outData(PortIndex port) override;
		//std::shared_ptr<QtNodeData> inData(PortIndex port);

		void setInData(std::shared_ptr<QtNodeData> data, PortIndex portIndex) override;

		bool tryInData(PortIndex portIndex, std::shared_ptr<QtNodeData> nodeData) override;

		NodeDataType dataType(PortType portType, PortIndex portIndex) const override;


		QWidget* embeddedWidget() override { return nullptr; }

		NodeValidationState validationState() const override;

		std::shared_ptr<Module> getModule();

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

		using OutFieldPtr = std::vector<std::shared_ptr<QtFieldData>>;
		//TODO: why weak_ptr?
		// using InFieldPtr = std::vector<std::weak_ptr<FieldData>>;
		
		using InFieldPtr = std::vector<std::shared_ptr<QtFieldData>>;
		InFieldPtr input_fields;
		OutFieldPtr output_fields;

		QString m_name;

		std::shared_ptr<Module> mModule = nullptr;

		NodeValidationState modelValidationState = NodeValidationState::Warning;
		QString modelValidationError = QString("Missing or incorrect inputs");

	private:

		FBase* getField(PortType portType, PortIndex portIndex) const;

		std::vector<FBase*>& getOutputFields();
		std::vector<FBase*>& getInputFields();

	private:
		bool mEditingEnabled = true;
	};
}
