#include "QtModuleWidget.h"

#include "Module.h"

#include "Common.h"

namespace Qt
{

	QtModuleWidget::QtModuleWidget(std::shared_ptr<Module> base)
	{
		mModule = base;

		if (mModule != nullptr)
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

		updateModule();
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

		return NodeDataType{ name.c_str(), name.c_str(), PortShape::Point };
	}


	std::shared_ptr<QtNodeData> QtModuleWidget::outData(PortIndex port)
	{
		return std::dynamic_pointer_cast<QtNodeData>(output_fields[port]);
	}


// 	std::shared_ptr<QtNodeData> QtModuleWidget::inData(PortIndex port)
// 	{
// 		// weak_ptr.lock() : if ptr is expired then return nullptr.
// 		// fprintf(stderr, (input_fields[port].expired()) ? "expired\n" : "nothing!\n");
// 		// return std::dynamic_pointer_cast<QtNodeData>(input_fields[port].lock());
// 		return std::dynamic_pointer_cast<QtNodeData>(input_fields[port]);
// 	}

	QString QtModuleWidget::caption() const
	{
// 		//	return m_name;
 		return dyno::FormatBlockCaptionName(mModule->caption());
	}

	QString QtModuleWidget::name() const
	{
		return QString::fromStdString(mModule->caption());
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
		if (!mEditingEnabled)
			return;

		auto fieldData = std::dynamic_pointer_cast<QtFieldData>(data);

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

	bool QtModuleWidget::tryInData(PortIndex portIndex, std::shared_ptr<QtNodeData> nodeData)
	{
		if (!mEditingEnabled)
			return false;

		try
		{
			auto& fieldExp = std::dynamic_pointer_cast<QtFieldData>(nodeData);
			if (fieldExp == nullptr)
				return false;

			auto fieldInp = input_fields[portIndex];

			if (fieldInp->getField()->getClassName() == fieldExp->getField()->getClassName())
			{
				std::string className = fieldInp->getField()->getClassName();
				if (className == dyno::InstanceBase::className())
				{
					return true;
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

		return true;
	}

	NodeValidationState QtModuleWidget::validationState() const
	{
		return modelValidationState;
	}

	std::shared_ptr<Module> QtModuleWidget::getModule()
	{
		return mModule;
	}

	void QtModuleWidget::enableEditing()
	{
		mEditingEnabled = true;
	}

	void QtModuleWidget::disableEditing()
	{
		mEditingEnabled = false;
	}

	QString QtModuleWidget::validationMessage() const
	{
		return modelValidationError;
	}

	void QtModuleWidget::updateModule()
	{
		bool hasAllInputs = mModule->isInputComplete();

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
		return portType == PortType::In ? mModule->getInputFields()[portIndex] : mModule->getOutputFields()[portIndex];
	}

	std::vector<FBase*>& QtModuleWidget::getOutputFields()
	{
		return mModule->getOutputFields();
	}

	std::vector<FBase*>& QtModuleWidget::getInputFields()
	{
		return mModule->getInputFields();
	}
}
