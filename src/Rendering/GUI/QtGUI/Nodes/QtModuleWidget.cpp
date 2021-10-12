#include "QtModuleWidget.h"

#include "FieldData.h"
#include "Module.h"

#include "../Common.h"

namespace QtNodes
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
				output_fields[i] = std::make_shared<FieldData>(outputs[i]);
			}

			//initialize in ports
			int input_num = getInputFields().size();
			input_fields.resize(input_num);
			auto inputs = getInputFields();
			for (int i = 0; i < inputs.size(); i++)
			{
				input_fields[i] = std::make_shared<FieldData>(inputs[i]);;
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


	BlockDataType QtModuleWidget::dataType(PortType portType, PortIndex portIndex) const
	{
		dyno::FBase* f = this->getField(portType, portIndex);

		std::string name = f->getClassName();

		return BlockDataType{ name.c_str(), name.c_str() };
	}


	std::shared_ptr<BlockData> QtModuleWidget::outData(PortIndex port)
	{
		return std::dynamic_pointer_cast<BlockData>(output_fields[port]);
	}


	std::shared_ptr<BlockData> QtModuleWidget::inData(PortIndex port)
	{
		// weak_ptr.lock() : if ptr is expired then return nullptr.
		// fprintf(stderr, (input_fields[port].expired()) ? "expired\n" : "nothing!\n");
		// return std::dynamic_pointer_cast<BlockData>(input_fields[port].lock());
		return std::dynamic_pointer_cast<BlockData>(input_fields[port]);
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

	void QtModuleWidget::setInData(std::shared_ptr<BlockData> data, PortIndex portIndex)
	{
		auto field_port = std::dynamic_pointer_cast<FieldData>(data);

		input_fields[portIndex] = field_port;

		if (field_port != nullptr)
		{
			auto in_fields = getInputFields();
			field_port->getField()->connectField(in_fields[portIndex]);
		}

		updateModule();
	}


	ValidationState QtModuleWidget::validationState() const
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
			modelValidationState = ValidationState::Valid;
			modelValidationError = QString();
		}
		else
		{
			modelValidationState = ValidationState::Warning;
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
