#include "QtModuleWidget.h"

#include "Module.h"

#include "Common.h"

namespace Qt
{

	QtModuleWidget::QtModuleWidget(Module* base)
	{
		m_module = base;

		if (m_module != nullptr)
		{
			//initialize out ports
			int output_num = getOutputFields().size();
			output_fields.resize(output_num);
			auto outputs = getOutputFields();
			for (int i = 0; i < outputs.size(); i++)
			{
				output_fields[i] = std::make_shared<QtFieldData>(outputs[i]);
			}

			//initialize in ports
			int input_num = getInputFields().size();
			input_fields.resize(input_num);
			auto inputs = getInputFields();
			for (int i = 0; i < inputs.size(); i++)
			{
				input_fields[i] = std::make_shared<QtFieldData>(inputs[i]);;
				// fprintf(stderr, (input_fields[i].expired()) ? "expired\n" : "nothing!\n");
			}
		}
	}

	unsigned int
		QtModuleWidget::nPorts(PortType portType) const
	{
		unsigned int result;

		if (portType == PortType::In)
		{
			result = input_fields.size();
		}
		else
		{
			result = output_fields.size();
		}

		return result;
	}


	NodeDataType QtModuleWidget::dataType(PortType portType, PortIndex portIndex) const
	{
		dyno::FBase* f = this->getField(portType, portIndex);

		std::string name = f->getClassName();

		return NodeDataType{ name.c_str(), name.c_str() };
	}


	std::shared_ptr<QtNodeData> QtModuleWidget::outData(PortIndex port)
	{
		return std::dynamic_pointer_cast<QtNodeData>(output_fields[port]);
	}


	std::shared_ptr<QtNodeData> QtModuleWidget::inData(PortIndex port)
	{
		// weak_ptr.lock() : if ptr is expired then return nullptr.
		// fprintf(stderr, (input_fields[port].expired()) ? "expired\n" : "nothing!\n");
		// return std::dynamic_pointer_cast<QtNodeData>(input_fields[port].lock());
		return std::dynamic_pointer_cast<QtNodeData>(input_fields[port]);
	}

	QString QtModuleWidget::caption() const
	{
		//	return m_name;
		std::string class_name = m_module->getClassInfo()->getClassName();
		if(class_name.find("VirtualModule") != std::string::npos) return QString::fromStdString(m_module->getName());
		else return QString::fromStdString(class_name);		
	}

	QString QtModuleWidget::name() const
	{
		std::string class_name = m_module->getClassInfo()->getClassName();
		if(class_name.find("VirtualModule") != std::string::npos) return QString::fromStdString(m_module->getName());
		else return QString::fromStdString(class_name);
	}

	bool QtModuleWidget::portCaptionVisible(PortType portType, PortIndex portIndex) const
	{
		Q_UNUSED(portType); Q_UNUSED(portIndex);
		return true;
	}

	QString QtModuleWidget::portCaption(PortType portType, PortIndex portIndex) const
	{
		dyno::FBase* f = this->getField(portType, portIndex);
		std::string name = f->getObjectName();

		return dyno::FormatBlockPortName(name);
	}

	void QtModuleWidget::setInData(std::shared_ptr<QtNodeData> data, PortIndex portIndex)
	{
		auto field_port = std::dynamic_pointer_cast<QtFieldData>(data);

		input_fields[portIndex] = field_port;

		if (field_port != nullptr)
		{
			auto in_fields = getInputFields();
			field_port->getField()->connect(in_fields[portIndex]);
		}

		updateModule();
	}


	NodeValidationState QtModuleWidget::validationState() const
	{
		return modelValidationState;
	}

	Module* QtModuleWidget::getModule()
	{
		return m_module;
	}

	QString QtModuleWidget::validationMessage() const
	{
		return modelValidationError;
	}

	void QtModuleWidget::updateModule()
	{
		bool hasAllInputs = true;

		for (int i = 0; i < input_fields.size(); i++)
		{

			//auto p = input_fields[i].lock();
			auto p = input_fields[i];

			hasAllInputs &= (p != nullptr);
		}

		if (hasAllInputs)
		{
			modelValidationState = NodeValidationState::Valid;
			modelValidationError = QString();
		}
		else
		{
			modelValidationState = NodeValidationState::Warning;
			modelValidationError = QStringLiteral("Missing or incorrect inputs");
		}

		for (int i = 0; i < output_fields.size(); i++)
		{
			Q_EMIT dataUpdated(i);
		}
	}

	FBase* QtModuleWidget::getField(PortType portType, PortIndex portIndex) const
	{
		return portType == PortType::In ? m_module->getInputFields()[portIndex] : m_module->getOutputFields()[portIndex];
	}

	std::vector<FBase*>& QtModuleWidget::getOutputFields()
	{
		return m_module->getOutputFields();
	}

	std::vector<FBase*>& QtModuleWidget::getInputFields()
	{
		return m_module->getInputFields();
	}
}
