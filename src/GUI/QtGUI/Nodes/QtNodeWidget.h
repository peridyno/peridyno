#pragma once

#include <QtCore/QObject>
#include <QtCore/QJsonObject>
#include <QtWidgets/QLabel>

#include "QtBlockDataModel.h"
#include "Framework/Node.h"

#include <iostream>


using dyno::Node;

using QtNodes::PortType;
using QtNodes::PortIndex;
using QtNodes::BlockData;
using QtNodes::BlockDataType;
using QtNodes::QtBlockDataModel;
using QtNodes::ValidationState;


class NodeImportData;
class NodeExportData;

namespace QtNodes
{
	/// The model dictates the number of inputs and outputs for the Node.
	/// In this example it has no logic.
	class QtNodeWidget : public QtBlockDataModel
	{
		Q_OBJECT

	public:
		QtNodeWidget(std::shared_ptr<Node> base = nullptr);

		virtual	~QtNodeWidget();

	public:

		QString caption() const override;

		QString name() const override;

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

		ConnectionPolicy portInConnectionPolicy(PortIndex portIndex) const override;

		std::shared_ptr<Node> getNode();

	protected:
		virtual void updateModule();

	protected:
		using ExportNodePtr = std::shared_ptr<NodeExportData>;
		using ImportNodePtr = std::vector<std::shared_ptr<NodeImportData>>;

		ImportNodePtr im_nodes;
		ExportNodePtr ex_node;

		std::shared_ptr<Node> m_node = nullptr;

		ValidationState modelValidationState = ValidationState::Valid;
		QString modelValidationError = QString("Missing or incorrect inputs");

	private:
	};


}
