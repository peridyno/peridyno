#include "WtModulelWidget.h"

WtModuleWidget::WtModuleWidget(std::shared_ptr<Module> base)
{
	m_Module = base;

	if (m_Module != nullptr)
	{
		//initialize out ports
		int output_num = getOutputFields().size();

		output_fields.resize(output_num);

		auto outputs = getOutputFields();

		for (int i = 0; i < outputs.size(); i++)
		{
			output_fields[i] = std::make_shared<WtFieldData>(outputs[i]);
		}

		//initialize in ports
		int input_num = getInputFields().size();

		input_fields.resize(input_num);

		auto inputs = getInputFields();

		for (int i = 0; i < inputs.size(); i++)
		{
			input_fields[i] = std::make_shared<WtFieldData>(inputs[i]);;
		}
	}

	updateModule();
}

std::string WtModuleWidget::caption() const
{
	return m_Module->caption();
}

bool WtModuleWidget::captionVisible() const
{
	return m_Module->captionVisible();
}

std::string WtModuleWidget::name() const
{
	return m_Module->caption();
}

std::string WtModuleWidget::portCaption(PortType portType, PortIndex portIndex) const
{
	dyno::FBase* f = this->getField(portType, portIndex);

	std::string name = f->getObjectName();

	return name;
}

std::string WtModuleWidget::nodeTips() const
{
	return m_Module->description();
}

std::string WtModuleWidget::portTips(PortType portType, PortIndex portIndex) const
{
	dyno::FBase* f = this->getField(portType, portIndex);

	auto fieldTip = [&](FBase* f) -> std::string {
		std::string tip;
		tip += "Class: " + f->getClassName() + "\n";
		tip += "Template: " + f->getTemplateName() + "\n";

		return tip;
		};

	return fieldTip(f);
}

std::string WtModuleWidget::validationMessage() const
{
	return modelValidationError;
}

unsigned int WtModuleWidget::nPorts(PortType portType) const
{
	return (portType == PortType::In) ? input_fields.size() : output_fields.size();
}

bool WtModuleWidget::portCaptionVisible(PortType portType, PortIndex portIndex) const
{
	return true;
}

std::shared_ptr<WtNodeData> WtModuleWidget::outData(PortIndex port)
{
	return std::dynamic_pointer_cast<WtNodeData>(output_fields[port]);
}

void WtModuleWidget::setInData(std::shared_ptr<WtNodeData> data, PortIndex portIndex)
{
	if (!mEditingEnabled)
		return;

	auto fieldData = std::dynamic_pointer_cast<WtFieldData>(data);

	if (fieldData != nullptr)
	{
		auto field = fieldData->getField();

		if (fieldData->connectionType() == CntType::Break)
		{
			field->disconnect(input_fields[portIndex]->getField());
			fieldData->setConnectionType(CntType::Link);
		}
		else
		{
			field->connect(input_fields[portIndex]->getField());
		}
	}

	updateModule();
}

bool WtModuleWidget::tryInData(PortIndex portIndex, std::shared_ptr<WtNodeData> nodeData)
{
	if (!mEditingEnabled)
		return false;

	try
	{
		auto fieldExp = std::dynamic_pointer_cast<WtFieldData>(nodeData);

		if (fieldExp == nullptr)
			return false;

		auto fieldInp = input_fields[portIndex];

		if (fieldInp->getField()->getClassName() == fieldExp->getField()->getClassName())
		{
			std::string className = fieldExp->getField()->getClassName();

			if (className == dyno::InstanceBase::className())
			{
				dyno::InstanceBase* instIn = dynamic_cast<dyno::InstanceBase*>(fieldInp->getField());

				dyno::InstanceBase* instOut = dynamic_cast<dyno::InstanceBase*>(fieldExp->getField());

				if (instIn != nullptr && instOut != nullptr)
					return instIn->canBeConnectedBy(instOut);

				return false;
			}
			else
			{
				return fieldInp->getField()->getTemplateName() == fieldExp->getField()->getTemplateName();
			}
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
	return true;
}

NodeDataType WtModuleWidget::dataType(PortType portType, PortIndex portIndex) const
{
	dyno::FBase* f = this->getField(portType, portIndex);

	std::string name = f->getClassName();

	return NodeDataType{ name.c_str(), name.c_str(), PortShape::Point };
}

NodeValidationState WtModuleWidget::validationState() const
{
	return modelValidationState;
}

std::shared_ptr<Module> WtModuleWidget::getModule()
{
	return m_Module;
}

void WtModuleWidget::enableEditing()
{
	mEditingEnabled = true;
}

void WtModuleWidget::disableEditing()
{
	mEditingEnabled = false;
}

void WtModuleWidget::updateModule()
{
	//bool hasAllInputs = m_Module->isInputComplete();

	//if (hasAllInputs)
	//{
	//	modelValidationState = NodeValidationState::Valid;
	//	modelValidationError = std::string();
	//}
	//else
	//{
	//	modelValidationState = NodeValidationState::Warning;
	//	//modelValidationError = QStringLiteral("Missing or incorrect inputs");
	//}

	//for (int i = 0; i < output_fields.size(); i++)
	//{
	//	dataUpdated(i);
	//}
}

FBase* WtModuleWidget::getField(PortType portType, PortIndex portIndex) const
{
	return portType == PortType::In ? m_Module->getInputFields()[portIndex] : m_Module->getOutputFields()[portIndex];
}

std::vector<FBase*>& WtModuleWidget::getOutputFields()
{
	return m_Module->getOutputFields();
}

std::vector<FBase*>& WtModuleWidget::getInputFields()
{
	return m_Module->getInputFields();
}