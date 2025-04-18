#include "WParameterDataNode.h"
#include <SceneGraph.h>

std::map<std::string, WParameterDataNode::FieldWidgetMeta> WParameterDataNode::sFieldWidgetMeta{};

WParameterDataNode::WParameterDataNode() 
	//:table(nullptr)
{
	FieldWidgetMeta WRealWidgetMeta
	{
		&typeid(float),
		WRealFieldWidget::WRealFieldWidgetConstructor
	};

	FieldWidgetMeta WVector3FieldWidgetMeta
	{
		&typeid(dyno::Vec3f),
		WVector3FieldWidget::WVector3FieldWidgetConstructor
	};

	FieldWidgetMeta WVector3dFieldWidgetMeta
	{
		&typeid(dyno::Vec3d),
		WVector3FieldWidget::WVector3FieldWidgetConstructor
	};

	FieldWidgetMeta WVector3iFieldWidgetMeta
	{
		&typeid(dyno::Vec3i),
		WVector3iFieldWidget::WVector3iFieldWidgetConstructor
	};

	FieldWidgetMeta WVector3uFieldWidgetMeta
	{
		&typeid(dyno::Vec3u),
		WVector3iFieldWidget::WVector3iFieldWidgetConstructor
	};

	FieldWidgetMeta WBoolFieldWidgetMeta
	{
		&typeid(bool),
		WBoolFieldWidget::WBoolFieldWidgetConstructor
	};

	FieldWidgetMeta WIntegerFieldWidgetMeta
	{
		&typeid(int),
		WIntegerFieldWidget::WIntegerFieldWidgetConstructor
	};

	FieldWidgetMeta WUIntegerFieldWidgetMeta
	{
		&typeid(dyno::uint),
		WUIntegerFieldWidget::WUIntegerFieldWidgetConstructor
	};

	FieldWidgetMeta WEnumFieldWidgetMeta
	{
		&typeid(dyno::PEnum),
		WEnumFieldWidget::WEnumFieldWidgetConstructor
	};

	FieldWidgetMeta WColorWidgetMeta
	{
		&typeid(dyno::Color),
		WColorWidget::WColorWidgetConstructor
	};

	FieldWidgetMeta WFileWidgetMeta
	{
		&typeid(dyno::FilePath),
		WFileWidget::WFileWidgetConstructor
	};

	registerWidget(WRealWidgetMeta);
	registerWidget(WVector3FieldWidgetMeta);
	registerWidget(WVector3dFieldWidgetMeta);
	registerWidget(WVector3iFieldWidgetMeta);
	registerWidget(WVector3uFieldWidgetMeta);
	registerWidget(WBoolFieldWidgetMeta);
	registerWidget(WIntegerFieldWidgetMeta);
	registerWidget(WUIntegerFieldWidgetMeta);
	registerWidget(WEnumFieldWidgetMeta);
	registerWidget(WColorWidgetMeta);
	registerWidget(WFileWidgetMeta);
}

WParameterDataNode::~WParameterDataNode()
{
}

void WParameterDataNode::setNode(std::shared_ptr<dyno::Node> node)
{
	mNode = node;
	layoutAboutToBeChanged().emit();
	layoutChanged().emit();
}

void WParameterDataNode::setModule(std::shared_ptr<dyno::Module> module)
{
	mModule = module;
	layoutAboutToBeChanged().emit();
	layoutChanged().emit();
}

int WParameterDataNode::columnCount(const Wt::WModelIndex& parent) const
{
	return 2;
}

int WParameterDataNode::rowCount(const Wt::WModelIndex& parent) const
{
	if (mNode != 0)
	{
		return mNode->getModuleList().size();
	}
	return 0;
}

Wt::cpp17::any WParameterDataNode::data(const Wt::WModelIndex& index, Wt::ItemDataRole role) const
{
	if (mNode != 0 && index.isValid())
	{
		auto mod = mNode->getModuleList();
		auto iter = mod.begin();
		std::advance(iter, index.row());

		if (role == Wt::ItemDataRole::Display || role == Wt::ItemDataRole::ToolTip)
		{
			if (index.column() == 0)
			{
				return (*iter)->getName();
			}
			if (index.column() == 1)
			{
				return (*iter)->getModuleType();
			}
		}
		else if (role == Wt::ItemDataRole::Decoration)
		{
			if (index.column() == 0)
			{
				return std::string("icons/module.png");
			}
		}
	}
	return Wt::cpp17::any();
}

Wt::cpp17::any WParameterDataNode::headerData(int section, Wt::Orientation orientation, Wt::ItemDataRole role) const
{
	if (orientation == Wt::Orientation::Horizontal && role == Wt::ItemDataRole::Display) {
		switch (section) {
		case 0:
			return std::string("Module");
		case 1:
			return std::string("Type");
		default:
			return Wt::cpp17::any();
		}
	}
	else
		return Wt::cpp17::any();
}

void WParameterDataNode::createParameterPanel(Wt::WContainerWidget* parameterWidget)
{
	parameterWidget->setMargin(0);
	//parameterWidget->setStyleClass("scrollable-content");
	auto layout = parameterWidget->setLayout(std::make_unique<Wt::WVBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);

	auto controlPanel = layout->addWidget(std::make_unique<Wt::WPanel>());
	controlPanel->setTitle("Control Variables");
	controlPanel->setCollapsible(true);
	controlPanel->setStyleClass("scrollable-content");
	controlPanel->setMargin(0);
	auto controlTable = controlPanel->setCentralWidget(std::make_unique<Wt::WTable>());

	auto statePanel = layout->addWidget(std::make_unique<Wt::WPanel>());
	statePanel->setTitle("State Variables");
	statePanel->setCollapsible(true);
	statePanel->setStyleClass("scrollable-content");
	statePanel->setMargin(0);
	auto stateTable = statePanel->setCentralWidget(std::make_unique<Wt::WTable>());

	std::vector<dyno::FBase*>& fields = mNode->getAllFields();
	int a = 0;
	for (dyno::FBase* var : fields)
	{
		if (var != nullptr)
		{
			if (var->getFieldType() == dyno::FieldTypeEnum::Param)
			{
				if (var->getClassName() == std::string("FVar"))
				{
					std::string template_name = var->getTemplateName();
					addScalarFieldWidget(controlTable, var->getObjectName(), var);
					Wt::log("info") << var->getTemplateName();
				}
			}
			else if (var->getFieldType() == dyno::FieldTypeEnum::State)
			{
				//Wt::log("info") << var->getDescription();
			}
		}
	}
	controlTable->setMargin(10);
}

void WParameterDataNode::createParameterPanelModule(Wt::WPanel* panel)
{
	auto table = panel->setCentralWidget(std::make_unique<Wt::WTable>());
	std::vector<dyno::FBase*>& fields = mModule->getAllFields();
	int a = 0;
	for (dyno::FBase* var : fields)
	{
		if (var != nullptr)
		{
			if (var->getFieldType() == dyno::FieldTypeEnum::Param)
			{
				if (var->getClassName() == std::string("FVar"))
				{
					std::string template_name = var->getTemplateName();
					addScalarFieldWidget(table, var->getObjectName(), var);
					Wt::log("info") << var->getTemplateName();
				}
			}
			else if (var->getFieldType() == dyno::FieldTypeEnum::State)
			{
				//addStateFieldWidget(var);
				Wt::log("info") << var->getDescription();
			}
		}
	}
	table->setMargin(10);
}

int WParameterDataNode::registerWidget(const FieldWidgetMeta& meta) {
	sFieldWidgetMeta[meta.type->name()] = meta;
	return 0;
}

WParameterDataNode::FieldWidgetMeta* WParameterDataNode::getRegistedWidget(const std::string& name)
{
	if (sFieldWidgetMeta.count(name))
		return &sFieldWidgetMeta.at(name);
	return nullptr;
}

Wt::WContainerWidget* WParameterDataNode::createFieldWidget(dyno::FBase* field)
{
	Wt::WContainerWidget* fw = nullptr;
	std::string template_name = field->getTemplateName();
	auto reg = getRegistedWidget(template_name);

	if (reg) {
		fw = reg->constructor(field);
	}

	return fw;
}

void WParameterDataNode::emit()
{
	changeValue_.emit(1);
}

void WParameterDataNode::castToDerived(Wt::WContainerWidget* fw)
{

	if (WRealFieldWidget* realWidget = dynamic_cast<WRealFieldWidget*>(fw))
	{
		realWidget->changeValue().connect(this, &WParameterDataNode::emit);
	}
	else if (WVector3FieldWidget* vec3fWidget = dynamic_cast<WVector3FieldWidget*>(fw))
	{
		vec3fWidget->changeValue().connect(this, &WParameterDataNode::emit);
	}
	else if (WVector3iFieldWidget* vec3iWidget = dynamic_cast<WVector3iFieldWidget*>(fw))
	{
		vec3iWidget->changeValue().connect(this, &WParameterDataNode::emit);
	}
	else if (WBoolFieldWidget* boolWidget = dynamic_cast<WBoolFieldWidget*>(fw))
	{
		boolWidget->changeValue().connect(this, &WParameterDataNode::emit);
	}
	else if (WIntegerFieldWidget* intWidget = dynamic_cast<WIntegerFieldWidget*>(fw))
	{
		intWidget->changeValue().connect(this, &WParameterDataNode::emit);
	}
	else if (WUIntegerFieldWidget* intuWidget = dynamic_cast<WUIntegerFieldWidget*>(fw))
	{
		intuWidget->changeValue().connect(this, &WParameterDataNode::emit);
	}
	else if (WEnumFieldWidget* enumWidget = dynamic_cast<WEnumFieldWidget*>(fw))
	{
		enumWidget->changeValue().connect(this, &WParameterDataNode::emit);
	}
	else if (WColorWidget* colorWidget = dynamic_cast<WColorWidget*>(fw))
	{
		colorWidget->changeValue().connect(this, &WParameterDataNode::emit);
	}
	else if (WFileWidget* fileWidget = dynamic_cast<WFileWidget*>(fw))
	{
		fileWidget->changeValue().connect(this, &WParameterDataNode::emit);
	}
	else
	{
		Wt::log("info") << "Error with dynamic_cast!";
	}

}

void WParameterDataNode::addScalarFieldWidget(Wt::WTable* table, std::string label, dyno::FBase* field, int labelWidth, int widgetWidth)
{
	Wt::WContainerWidget* fw = createFieldWidget(field);
	castToDerived(fw);
	if (fw)
	{
		std::unique_ptr<Wt::WContainerWidget> mWidget(fw);
		if (fw != nullptr) {
			int row = table->rowCount();
			auto cell0 = table->elementAt(row, 0);
			auto cell1 = table->elementAt(row, 1);

			cell0->addNew<Wt::WText>(label);
			cell0->setContentAlignment(Wt::AlignmentFlag::Middle);
			cell0->setWidth(labelWidth);

			cell1->setContentAlignment(Wt::AlignmentFlag::Middle);
			cell1->setWidth(widgetWidth);

			cell1->addWidget(std::unique_ptr<Wt::WContainerWidget>(std::move(mWidget)));
		}
	}
}

void WParameterDataNode::addStateFieldWidget(dyno::FBase* field)
{

}
