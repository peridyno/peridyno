#include "WtNodeWidget.h"

#include "Node.h"
#include "NodePort.h"

//#include "Format.h"

#include "FInstance.h"
#include "Field.h"

WtNodeWidget::WtNodeWidget(std::shared_ptr<Node> base)
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
			mNodeInport[i] = std::make_shared<WtImportNode>(inputs[i]);
		}

		//initialize out node ports
		mNodeExport = std::make_shared<WtExportNode>(base);

		int output_fnum = getOutputFields().size();
		mFieldExport.resize(output_fnum);
		auto fOutputs = getOutputFields();
		for (int i = 0; i < fOutputs.size(); i++)
		{
			mFieldExport[i] = std::make_shared<WtFieldData>(fOutputs[i]);
		}

		//initialize in ports
		int input_fnum = getInputFields().size();
		mFieldInport.resize(input_fnum);
		auto fInputs = getInputFields();
		for (int i = 0; i < fInputs.size(); i++)
		{
			mFieldInport[i] = std::make_shared<WtFieldData>(fInputs[i]);;
		}
	}
}

WtNodeWidget::~WtNodeWidget() {}

unsigned int WtNodeWidget::nPorts(PortType portType) const
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

NodeDataType WtNodeWidget::dataType(PortType portType, PortIndex portIndex) const
{
	switch (portType)
	{
	case PortType::In:
		if (portIndex < mNodeInport.size()) {
			//TODO: return more accurate description
			return NodeDataType{ "port", "port", PortShape::Bullet };
		}
		else {
			auto& inputFields = this->getInputFields();
			std::string str = inputFields[portIndex - mNodeInport.size()]->getClassName();

			return NodeDataType{ str.c_str(), str.c_str(), PortShape::Point };
		}
		break;

	case PortType::Out:
		if (portIndex == 0) {
			//TODO: return more accurate description
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

std::shared_ptr<WtNodeData> WtNodeWidget::outData(PortIndex port)
{


	return port == 0 ? std::static_pointer_cast<WtNodeData>(mNodeExport) : std::static_pointer_cast<WtNodeData>(mFieldExport[port - 1]);
}

std::string WtNodeWidget::caption() const
{
	return mNode->caption();
}

std::string WtNodeWidget::name() const
{
	return mNode->caption();
	//return mNode->getClassInfo()->getClassName();
}

std::string WtNodeWidget::nodeTips() const
{
	return mNode->description();
}

bool WtNodeWidget::portCaptionVisible(PortType portType, PortIndex portIndex) const
{
	(void)portType;; (void)portIndex;;
	return true;
}

std::string WtNodeWidget::portCaption(PortType portType, PortIndex portIndex) const
{
	switch (portType)
	{
	case PortType::In:
		if (portIndex < mNodeInport.size()) {
			return mNode->getImportNodes()[portIndex]->getPortName();
		}
		else {
			auto& inputFields = this->getInputFields();

			return inputFields[portIndex - mNodeInport.size()]->getObjectName();
		}
		break;

	case PortType::Out:
		if (portIndex == 0) {
			//return dyno::FormatBlockPortName(mNode->getClassInfo()->getClassName());
			return "";
		}
		else {
			auto& outputFields = this->getOutputFields();

			return outputFields[portIndex - 1]->getObjectName();
		}
		break;

	case PortType::None:
		break;
	}
}

std::string WtNodeWidget::portTips(PortType portType, PortIndex portIndex) const
{
	std::string tip;

	auto nodeTip = [&](Node* node) -> std::string {
		return node->getClassInfo()->getClassName();
		};

	auto fieldTip = [&](FBase* f) -> std::string {
		tip += "Class: " + f->getClassName() + "\n";
		tip += "Template: " + f->getTemplateName() + "\n";

		return tip;
		};

	switch (portType)
	{
	case PortType::In:
		if (portIndex < mNodeInport.size()) {
			return mNode->getImportNodes()[portIndex]->getPortName();
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

void WtNodeWidget::setInData(std::shared_ptr<WtNodeData> data, PortIndex portIndex)
{
	if (!mEditingEnabled)
		return;

	if (portIndex < mNodeInport.size())
	{
		auto node_port = std::dynamic_pointer_cast<WtExportNode>(data);

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
		auto fieldData = std::dynamic_pointer_cast<WtFieldData>(data);

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

bool WtNodeWidget::tryInData(PortIndex portIndex, std::shared_ptr<WtNodeData> nodeData)
{
	if (!mEditingEnabled)
		return false;

	if (portIndex < mNodeInport.size())
	{
		try
		{
			auto nodeExp = std::dynamic_pointer_cast<WtExportNode>(nodeData);

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
			auto fieldExp = std::dynamic_pointer_cast<WtFieldData>(nodeData);
			if (fieldExp == nullptr)
				return false;

			auto fieldInp = mFieldInport[portIndex - mNodeInport.size()];

			if (fieldInp->getField()->getClassName() == fieldExp->getField()->getClassName())
			{
				std::string className = fieldInp->getField()->getClassName();
				if (className == dyno::InstanceBase::className())
				{
					dyno::InstanceBase* instIn = dynamic_cast<dyno::InstanceBase*>(fieldInp->getField());
					dyno::InstanceBase* instOut = dynamic_cast<dyno::InstanceBase*>(fieldExp->getField());

					if (instIn != nullptr && instOut != nullptr)
						return instIn->canBeConnectedBy(instOut);

					return false;
				}
				else
					return fieldInp->getField()->getTemplateName() == fieldExp->getField()->getTemplateName();
			}
			else
			{
				return false;
			}
		}
		catch (std::bad_cast)
		{
			return false;
		}
	}
}

NodeValidationState WtNodeWidget::validationState() const
{
	return modelValidationState;
}

WtNodeDataModel::ConnectionPolicy WtNodeWidget::portInConnectionPolicy(PortIndex portIndex) const
{
	if (portIndex < mNodeInport.size())
	{
		auto portType = mNodeInport[portIndex]->getNodePort()->getPortType();

		return portType == dyno::NodePortType::Single ? ConnectionPolicy::One : ConnectionPolicy::Many;
	}
	else
	{
		return ConnectionPolicy::One;
	}
}

std::shared_ptr<Node> WtNodeWidget::getNode()
{
	return mNode;
}

std::string WtNodeWidget::validationMessage() const
{
	return modelValidationError;
}

void WtNodeWidget::updateModule()
{
	modelValidationState = NodeValidationState::Valid;
}

std::vector<FBase*>& WtNodeWidget::getOutputFields() const
{
	return mNode->getOutputFields();
}

std::vector<FBase*>& WtNodeWidget::getInputFields() const
{
	return mNode->getInputFields();
}

void WtNodeWidget::enableEditing()
{
	mEditingEnabled = true;
}

void WtNodeWidget::disableEditing()
{
	mEditingEnabled = false;
}