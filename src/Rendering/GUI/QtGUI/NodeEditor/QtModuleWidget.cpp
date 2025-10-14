#include "QtModuleWidget.h"

#include "Module.h"

#include "Format.h"

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

			mModuleExport = std::make_shared<QtExportModule>(base);

			//initialize in ports
			int input_num = getInputFields().size();
			input_fields.resize(input_num);
			auto inputs = getInputFields();
			for (int i = 0; i < inputs.size(); i++)
			{
				input_fields[i] = std::make_shared<QtFieldData>(inputs[i]);;
				// fprintf(stderr, (input_fields[i].expired()) ? "expired\n" : "nothing!\n");
			}

			auto imports = mModule->getImportModules();
			auto imports_num = imports.size();

			mModuleImports.resize(imports_num);
			for (int i = 0; i < imports.size(); i++)
			{
				mModuleImports[i] = std::make_shared<QtImportModule>(imports[i]);
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
			result = mModule->allowImported() ? mModuleImports.size() + input_fields.size() : input_fields.size();
		}
		else
		{
			result = mModule->allowExported() ? 1 + output_fields.size() : output_fields.size();
		}

		return result;
	}


	NodeDataType QtModuleWidget::dataType(PortType portType, PortIndex portIndex) const
	{
		switch (portType)
		{
		case PortType::In:
			if (mModule->allowImported() && portIndex < mModuleImports.size())
			{
				return NodeDataType{ "port", "port", PortShape::Arrow };
			}
			else
			{
				PortIndex recalculatedIndex = mModule->allowImported() ? portIndex - mModuleImports.size() : portIndex;

				dyno::FBase* f = this->getField(portType, recalculatedIndex);

				std::string name = f->getClassName();

				return NodeDataType{ name.c_str(), name.c_str(), PortShape::Point };
			}
			break;
		case PortType::Out:
			if (mModule->allowExported() && portIndex == 0)
			{
				return NodeDataType{ "port", "port", PortShape::Arrow };
			}
			else
			{
				PortIndex recalculatedIndex = mModule->allowExported() ? portIndex - 1 : portIndex;

				dyno::FBase* f = this->getField(portType, recalculatedIndex);

				std::string name = f->getClassName();

				return NodeDataType{ name.c_str(), name.c_str(), PortShape::Point };
			}
			break;
		case PortType::None:
			break;
		}

		return NodeDataType{ "port", "port", PortShape::Point };
	}


	std::shared_ptr<QtNodeData> QtModuleWidget::outData(PortIndex port)
	{
		if (mModule->allowExported())
		{
			return port == 0 ? std::static_pointer_cast<QtNodeData>(mModuleExport) : std::static_pointer_cast<QtNodeData>(output_fields[port - 1]);
		}
		else
			return std::static_pointer_cast<QtNodeData>(output_fields[port]);
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

	bool QtModuleWidget::captionVisible() const
	{
		return mModule->captionVisible();
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
		switch (portType)
		{
		case PortType::In:
			if (mModule->allowImported() && portIndex < mModuleImports.size())
			{
				return QString("");
			}
			else
			{
				PortIndex recalculatedIndex = mModule->allowImported() ? portIndex - mModuleImports.size() : portIndex;
				dyno::FBase* f = this->getField(portType, recalculatedIndex);
				std::string name = f->getObjectName();

				return dyno::FormatBlockPortName(name);
			}
			break;
		case PortType::Out:
			if (mModule->allowExported() && portIndex == 0)
			{
				return QString("");
			}
			else
			{
				PortIndex recalculatedIndex = mModule->allowExported() ? portIndex - 1 : portIndex;
				dyno::FBase* f = this->getField(portType, recalculatedIndex);
				std::string name = f->getObjectName();

				return dyno::FormatBlockPortName(name);
			}
			break;
		case PortType::None:
			break;
		}

		return QString("");
	}

	QString QtModuleWidget::nodeTips() const
	{
		return dyno::FormatDescription(mModule->description());
	}

	QString QtModuleWidget::portTips(PortType portType, PortIndex portIndex) const
	{
		auto fieldTip = [&](FBase* f) -> QString {
			std::string tip;
			tip += "Class: " + f->getClassName() + "\n";
			tip += "Template: " + f->getTemplateName() + "\n";

			return QString::fromStdString(tip);
			};

		switch (portType)
		{
		case PortType::In:
			if (mModule->allowImported() && portIndex < mModuleImports.size())
			{
				return QString("");
			}
			else
			{
				PortIndex recalculatedIndex = mModule->allowImported() ? portIndex - mModuleImports.size() : portIndex;

				dyno::FBase* f = this->getField(portType, recalculatedIndex);

				return fieldTip(f);
			}
			break;
		case PortType::Out:
			if (mModule->allowExported() && portIndex == 0)
			{
				return QString("");
			}
			else
			{
				PortIndex recalculatedIndex = mModule->allowExported() ? portIndex - 1 : portIndex;
				dyno::FBase* f = this->getField(portType, recalculatedIndex);

				return fieldTip(f);
			}
			break;
		case PortType::None:
			break;
		}

		return QString("");
	}

	void QtModuleWidget::setInData(std::shared_ptr<QtNodeData> data, PortIndex portIndex)
	{
		if (!mEditingEnabled)
			return;

		if (this->allowImported() && portIndex < mModuleImports.size())
		{
			auto module_port = std::dynamic_pointer_cast<QtExportModule>(data);

			if (module_port != nullptr)
			{
				auto m = module_port->getModule();

				if (module_port->connectionType() == CntType::Break)
				{
					//mNodeInport[portIndex]->getNodePort()->removeNode(nd.get());
					m->disconnect(mModuleImports[portIndex]->getModulePort());

					//TODO: recover the connection state, use a more elegant way in the future
					data->setConnectionType(CntType::Link);
				}
				else
				{
					m->connect(mModuleImports[portIndex]->getModulePort());
				}
			}
		}
		else
		{
			PortIndex recalcualtedIndex = this->allowImported() ? portIndex - mModuleImports.size() : portIndex;
			auto fieldData = std::dynamic_pointer_cast<QtFieldData>(data);

			if (fieldData != nullptr)
			{
				auto field = fieldData->getField();

				if (fieldData->connectionType() == CntType::Break)
				{
					field->disconnect(input_fields[recalcualtedIndex]->getField());
					fieldData->setConnectionType(CntType::Link);
				}
				else
				{
					field->connect(input_fields[recalcualtedIndex]->getField());
				}
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
			if (this->allowImported() && portIndex < mModuleImports.size())
			{
				try
				{
					auto moduleExp = std::dynamic_pointer_cast<QtExportModule>(nodeData);

					if (moduleExp == nullptr)
						return false;

					auto moduleImp = mModuleImports[portIndex];

					return moduleImp->getModulePort()->isKindOf(moduleExp->getModule().get());;
				}
				catch (std::bad_cast)
				{
					return false;
				}
			}
			else
			{
				PortIndex recalcualtedIndex = this->allowImported() ? portIndex - mModuleImports.size() : portIndex;

				auto fieldExp = std::dynamic_pointer_cast<QtFieldData>(nodeData);
				if (fieldExp == nullptr)
					return false;

				auto fieldInp = input_fields[recalcualtedIndex];

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

	QtNodeDataModel::ConnectionPolicy QtModuleWidget::portInConnectionPolicy(PortIndex portIndex) const
	{
		if (portIndex < mModuleImports.size())
		{
			auto portType = mModuleImports[portIndex]->getModulePort()->getPortType();

			return portType == dyno::ModulePortType::M_Single ? ConnectionPolicy::One : ConnectionPolicy::Many;
		}
		else
		{
			auto fieldInp = input_fields[portIndex - mModuleImports.size()];

			return fieldInp->getField()->inputPolicy() == FBase::One ? ConnectionPolicy::One : ConnectionPolicy::Many;
		}
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
