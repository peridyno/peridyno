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
			result = (unsigned int)m_node->getAllNodePorts().size();
		}
		else
		{
			result = 1;
		}

		return result;
	}

	NodeDataType QtNodeWidget::dataType(PortType portType, PortIndex portIndex) const
	{
		return NodeDataType{ "port", "port", PortShape::Bullet };
	}


	std::shared_ptr<QtNodeData>
		QtNodeWidget::outData(PortIndex port)
	{
		return std::static_pointer_cast<QtNodeData>(ex_node);
	}


	std::shared_ptr<QtNodeData> QtNodeWidget::inData(PortIndex port)
	{
		return std::static_pointer_cast<QtNodeData>(im_nodes[port]);
	}

	QString QtNodeWidget::caption() const
	{
		return QString::fromStdString(m_node->getClassInfo()->getClassName());
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
		if (portType == PortType::In)
		{
			return dyno::FormatBlockPortName(m_node->getAllNodePorts()[portIndex]->getPortName());
		}
		else
		{
			return QString::fromStdString(m_node->getClassInfo()->getClassName());
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

}
