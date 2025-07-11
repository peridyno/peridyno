#include "QtNodeWidget.h"

#include "nodes/QConnection"
#include "nodes/QNode"

#include "Node.h"
#include "NodePort.h"

#include "Format.h"

#include "FInstance.h"
#include "Field.h"

namespace Qt
{
	QtNodeWidget::QtNodeWidget(std::shared_ptr<Node> base)
	{
		mNode = base;

		if (mNode != nullptr)
		{
			//initialize in node ports
			auto inputs = mNode->getImportNodes();
			auto input_num = inputs.size();

			mNodeInport.resize(input_num);
			for (int i = 0; i < inputs.size(); i++)
			{
				mNodeInport[i] = std::make_shared<QtImportNode>(inputs[i]);
			}

			//initialize out node ports
			mNodeExport = std::make_shared<QtExportNode>(base);

			int output_fnum = getOutputFields().size();
			mFieldExport.resize(output_fnum);
			auto fOutputs = getOutputFields();
			for (int i = 0; i < fOutputs.size(); i++)
			{
				mFieldExport[i] = std::make_shared<QtFieldData>(fOutputs[i]);
			}

			//initialize in ports
			int input_fnum = getInputFields().size();
			mFieldInport.resize(input_fnum);
			auto fInputs = getInputFields();
			for (int i = 0; i < fInputs.size(); i++)
			{
				mFieldInport[i] = std::make_shared<QtFieldData>(fInputs[i]);
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
			result = (unsigned int)mNode->getImportNodes().size() + mFieldInport.size();
		}
		else
		{
			result = 1 + mFieldExport.size();
		}

		return result;
	}

	NodeDataType QtNodeWidget::dataType(PortType portType, PortIndex portIndex) const
	{
		switch (portType)
		{
		case PortType::In:
			if (portIndex < mNodeInport.size()) {
				//TODO: return more accurate description
				return NodeDataType{ "port", "port", PortShape::Diamond };
			}
			else {
				auto& inputFields = this->getInputFields();
				std::string str = inputFields[portIndex - mNodeInport.size()]->getClassName();

				if (str == "FInstance" || str == "FInstances")
				{
					std::string insName = dyno::InstanceBase::className();
					return NodeDataType{ insName.c_str(), insName.c_str(), PortShape::Point };
				}
				else
					return NodeDataType{ str.c_str(), str.c_str(), PortShape::Point };

			}
			break;

		case PortType::Out:
			if (portIndex == 0) {
				//TODO: return more accurate description
				return NodeDataType{ "port", "port", PortShape::Diamond };
			}
			else {
				auto& outputFields = this->getOutputFields();
				std::string str = outputFields[portIndex - 1]->getClassName();

				if (str == "FInstance" || str == "FInstances")
				{
					std::string insName = dyno::InstanceBase::className();
					return NodeDataType{ insName.c_str(), insName.c_str(), PortShape::Point };
				}
				else
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
		return port == 0 ? std::static_pointer_cast<QtNodeData>(mNodeExport) : std::static_pointer_cast<QtNodeData>(mFieldExport[port - 1]);
	}

// 	std::shared_ptr<QtNodeData> QtNodeWidget::inData(PortIndex port)
// 	{
// 		return port < mNodeInport.size() ? std::static_pointer_cast<QtNodeData>(mNodeInport[port]) : std::static_pointer_cast<QtNodeData>(mFieldInport[port - mNodeInport.size()]);
// 	}

	QString QtNodeWidget::caption() const
	{
		return dyno::FormatBlockCaptionName(mNode->caption());
	}

	QString QtNodeWidget::name() const
	{
		return dyno::FormatBlockCaptionName(mNode->caption());
		//return QString::fromStdString(mNode->getClassInfo()->getClassName());
	}

	QString QtNodeWidget::nodeTips() const
	{
		return dyno::FormatDescription(mNode->description());
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
			if (portIndex < mNodeInport.size()) {
				return dyno::FormatBlockPortName(mNode->getImportNodes()[portIndex]->getPortName());
			}
			else {
				auto& inputFields = this->getInputFields();

				return dyno::FormatBlockPortName(inputFields[portIndex - mNodeInport.size()]->getObjectName());
			}
			break;

		case PortType::Out:
			if (portIndex == 0) {
				//return dyno::FormatBlockPortName(mNode->getClassInfo()->getClassName());
				return dyno::FormatBlockPortName("");
			}
			else {
				auto& outputFields = this->getOutputFields();

				return dyno::FormatBlockPortName(outputFields[portIndex - 1]->getObjectName());
			}
			break;

		case PortType::None:
			break;
		}

		return QString("");
	}

	QString QtNodeWidget::portTips(PortType portType, PortIndex portIndex) const
	{
		std::string tip;

		auto nodeTip = [&](Node* node) -> QString {
			return QString::fromStdString(node->getClassInfo()->getClassName());
		};

		auto fieldTip = [&](FBase* f) -> QString {
			tip += "Class: " + f->getClassName() + "\n";
			tip += "Template: " + f->getTemplateName() + "\n";

			return QString::fromStdString(tip);
		};

		switch (portType)
		{
		case PortType::In:
			if (portIndex < mNodeInport.size()) {
				return dyno::FormatBlockPortName(mNode->getImportNodes()[portIndex]->getPortName());
			}
			else {
				auto& inputFields = this->getInputFields();
				return fieldTip(inputFields[portIndex - mNodeInport.size()]);
			}
			break;

		case PortType::Out:
			if (portIndex == 0) {
				return nodeTip(mNode.get());
			}
			else {
				auto& outputFields = this->getOutputFields();
				return fieldTip(outputFields[portIndex - 1]);
			}

			break;

		case PortType::None:
			break;
		}
	}

	void QtNodeWidget::setInData(std::shared_ptr<QtNodeData> data, PortIndex portIndex)
	{
		if (!mEditingEnabled)
			return;

		if (portIndex < mNodeInport.size())
		{
			auto node_port = std::dynamic_pointer_cast<QtExportNode>(data);

			if (node_port != nullptr)
			{
				auto nd = node_port->getNode();

				if (node_port->connectionType() == CntType::Break)
				{
					//mNodeInport[portIndex]->getNodePort()->removeNode(nd.get());
					nd->disconnect(mNodeInport[portIndex]->getNodePort());

					//TODO: recover the connection state, use a more elegant way in the future
					data->setConnectionType(CntType::Link);
				}
				else
				{
					//mNodeInport[portIndex]->getNodePort()->addNode(nd.get());
					nd->connect(mNodeInport[portIndex]->getNodePort());
				}
			}
		}
		else
		{
			auto fieldData = std::dynamic_pointer_cast<QtFieldData>(data);

			if (fieldData != nullptr)
			{
				auto field = fieldData->getField();

				if (fieldData->connectionType() == CntType::Break)
				{
					field->disconnect(mFieldInport[portIndex - mNodeInport.size()]->getField());
					fieldData->setConnectionType(CntType::Link);
				}
				else
				{ 
					field->connect(mFieldInport[portIndex - mNodeInport.size()]->getField());
				}
			}
		}

		updateModule();
	}


	bool QtNodeWidget::tryInData(PortIndex portIndex, std::shared_ptr<QtNodeData> nodeData)
	{
		if (!mEditingEnabled)
			return false;

		if (portIndex < mNodeInport.size())
		{
			try
			{
				auto nodeExp = std::dynamic_pointer_cast<QtExportNode>(nodeData);

				if (nodeExp == nullptr)
					return false;

				auto nodeInp = mNodeInport[portIndex];

				return nodeInp->getNodePort()->isKindOf(nodeExp->getNode().get());;
			}
			catch (std::bad_cast)
			{
				return false;
			}
		}
		else
		{
			try
			{
				auto fieldExp = std::dynamic_pointer_cast<QtFieldData>(nodeData);
				if (fieldExp == nullptr)
					return false;

				auto fieldInp = mFieldInport[portIndex - mNodeInport.size()];

				auto fIn = fieldInp->getField();
				auto fExp = fieldExp->getField();

				dyno::InstanceBase* instIn = dynamic_cast<dyno::InstanceBase*>(fIn);
				dyno::InstanceBase* instOut = dynamic_cast<dyno::InstanceBase*>(fExp);

				if (instIn != nullptr && instOut != nullptr)
					return instIn->canBeConnectedBy(instOut);
				else if (instIn == nullptr && instOut == nullptr)
				{
					return fIn->getClassName() == fExp->getClassName() && fIn->getTemplateName() == fExp->getTemplateName();
				}
				else
					return false;

			}
			catch (std::bad_cast)
			{
				return false;
			}
		}
	}

	NodeValidationState QtNodeWidget::validationState() const
	{
		return modelValidationState;
	}

	QtNodeDataModel::ConnectionPolicy QtNodeWidget::portInConnectionPolicy(PortIndex portIndex) const
	{
		if (portIndex < mNodeInport.size())
		{
			auto portType = mNodeInport[portIndex]->getNodePort()->getPortType();

			return portType == dyno::NodePortType::Single ? ConnectionPolicy::One : ConnectionPolicy::Many;
		}
		else
		{
			auto fieldInp = mFieldInport[portIndex - mNodeInport.size()];

			return fieldInp->getField()->inputPolicy() == FBase::One ? ConnectionPolicy::One : ConnectionPolicy::Many;
		}
	}

	std::shared_ptr<Node> QtNodeWidget::getNode()
	{
		return mNode;
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
		return mNode->getOutputFields();
	}

	std::vector<FBase*>& QtNodeWidget::getInputFields() const
	{
		return mNode->getInputFields();
	}

	void QtNodeWidget::enableEditing()
	{
		mEditingEnabled = true;
	}

	void QtNodeWidget::disableEditing()
	{
		mEditingEnabled = false;
	}
}
