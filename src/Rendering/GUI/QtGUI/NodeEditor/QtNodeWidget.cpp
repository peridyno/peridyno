#include "QtNodeWidget.h"

#include "Node.h"
#include "NodePort.h"

#include "Common.h"

namespace Qt
{
	QtNodeWidget::QtNodeWidget(std::shared_ptr<Node> base)
	{
		m_node = base;

		if (m_node != nullptr)
		{
			//initialize in node ports
			auto inputs = m_node->getAllNodePorts();
			auto input_num = inputs.size();

			im_nodes.resize(input_num);
			for (int i = 0; i < inputs.size(); i++)
			{
				im_nodes[i] = std::make_shared<QtNodeImportData>(inputs[i]);
			}

			//initialize out node ports
			ex_node = std::make_shared<QtNodeExportData>(base);

			int output_fnum = getOutputFields().size();
			output_fields.resize(output_fnum);
			auto fOutputs = getOutputFields();
			for (int i = 0; i < fOutputs.size(); i++)
			{
				output_fields[i] = std::make_shared<QtFieldData>(fOutputs[i]);
			}

			//initialize in ports
			int input_fnum = getInputFields().size();
			input_fields.resize(input_fnum);
			auto fInputs = getInputFields();
			for (int i = 0; i < fInputs.size(); i++)
			{
				input_fields[i] = std::make_shared<QtFieldData>(fInputs[i]);;
			}
		}

	}

	QtNodeWidget::~QtNodeWidget()
	{
	}

	unsigned int
		QtNodeWidget::nPorts(PortType portType) const
	{
		unsigned int result;

		if (portType == PortType::In)
		{
			result = (unsigned int)m_node->getAllNodePorts().size() + input_fields.size();
		}
		else
		{
			result = 1 + output_fields.size();
		}

		return result;
	}

	NodeDataType QtNodeWidget::dataType(PortType portType, PortIndex portIndex) const
	{
		switch (portType)
		{
		case PortType::In:
			if (portIndex < im_nodes.size()) {
				return NodeDataType{ "port", "port", PortShape::Bullet };
			}
			else {
				auto& inputFields = this->getInputFields();
				std::string str = inputFields[portIndex - im_nodes.size()]->getClassName();

				return NodeDataType{ str.c_str(), str.c_str(), PortShape::Point };
			}
			break;

		case PortType::Out:
			if (portIndex == 0) {
				return NodeDataType{ "port", "port", PortShape::Bullet };
			}
			else {
				auto& outputFields = this->getOutputFields();
				std::string str = outputFields[portIndex - 1]->getClassName();

				return NodeDataType{ str.c_str(), str.c_str(), PortShape::Point };
			}
			break;

		case PortType::None:
			break;
		}

		return NodeDataType{ "port", "port", PortShape::Point };
	}

	std::shared_ptr<QtNodeData>
		QtNodeWidget::outData(PortIndex port)
	{
		return port == 0 ? std::static_pointer_cast<QtNodeData>(ex_node) : std::static_pointer_cast<QtNodeData>(output_fields[port - 1]);
	}

	std::shared_ptr<QtNodeData> QtNodeWidget::inData(PortIndex port)
	{
		return port < im_nodes.size() ? std::static_pointer_cast<QtNodeData>(im_nodes[port]) : std::static_pointer_cast<QtNodeData>(input_fields[port - im_nodes.size()]);
	}

	QString QtNodeWidget::caption() const
	{
		return dyno::FormatBlockPortName(m_node->getClassInfo()->getClassName());
	}

	QString QtNodeWidget::name() const
	{
		return QString::fromStdString(m_node->getClassInfo()->getClassName());
	}

	bool QtNodeWidget::portCaptionVisible(PortType portType, PortIndex portIndex) const
	{
		Q_UNUSED(portType); Q_UNUSED(portIndex);
		return true;
	}

	QString QtNodeWidget::portCaption(PortType portType, PortIndex portIndex) const
	{
		switch (portType)
		{
		case PortType::In:
			if (portIndex < im_nodes.size()) {
				return dyno::FormatBlockPortName(m_node->getAllNodePorts()[portIndex]->getPortName());
			}
			else {
				auto& inputFields = this->getInputFields();

				return dyno::FormatBlockPortName(inputFields[portIndex - im_nodes.size()]->getObjectName());
			}
			break;

		case PortType::Out:
			if (portIndex == 0) {
				return dyno::FormatBlockPortName(m_node->getClassInfo()->getClassName());
			}
			else {
				auto& outputFields = this->getOutputFields();

				return dyno::FormatBlockPortName(outputFields[portIndex - 1]->getObjectName());
			}
			break;

		case PortType::None:
			break;
		}
	}

	void QtNodeWidget::setInData(std::shared_ptr<QtNodeData> data, PortIndex portIndex)
	{
		auto node_port = std::dynamic_pointer_cast<QtNodeExportData>(data);

		if (node_port != nullptr)
		{
			auto nd = node_port->getNode();

			if (node_port->isToDisconnected())
			{
				im_nodes[portIndex]->getNodePort()->removeNode(nd);
				node_port->setDisconnected(false);
			}
			else
			{
				im_nodes[portIndex]->getNodePort()->addNode(nd);
			}
		}

		updateModule();
	}


	NodeValidationState QtNodeWidget::validationState() const
	{
		return modelValidationState;
	}

	QtNodeDataModel::ConnectionPolicy QtNodeWidget::portInConnectionPolicy(PortIndex portIndex) const
	{
		return ConnectionPolicy::Many;
	}

	std::shared_ptr<Node> QtNodeWidget::getNode()
	{
		return m_node;
	}

	QString QtNodeWidget::validationMessage() const
	{
		return modelValidationError;
	}

	void QtNodeWidget::updateModule()
	{
		modelValidationState = NodeValidationState::Valid;
	}


	std::vector<FBase*>& QtNodeWidget::getOutputFields() const
	{
		return m_node->getOutputFields();
	}

	std::vector<FBase*>& QtNodeWidget::getInputFields() const
	{
		return m_node->getInputFields();
	}
}
