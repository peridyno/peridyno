#include "WParameterDataNode.h"
#include <SceneGraph.h>

std::map<std::string, WParameterDataNode::FieldWidgetMeta> WParameterDataNode::sFieldWidgetMeta{};

WParameterDataNode::WParameterDataNode() :table(nullptr)
{
	FieldWidgetMeta WRealWidgetMeta
	{
		&typeid(float),
		WRealFieldWidget::WRealFieldWidgetConstructor
	};

	registerWidget(WRealWidgetMeta);
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

void WParameterDataNode::createParameterPanel(Wt::WPanel* panel)
{
	table = panel->setCentralWidget(std::make_unique<Wt::WTable>());
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
					if (template_name == "float")
					{
						//auto test = addTableNodeRow<WRealFieldWidget>(table, var->getObjectName(), var);
						//test->setValue();
						addScalarFieldWidget(table, var->getObjectName(), var);
					}
					//int row = table->rowCount();
					//auto cell0 = table->elementAt(row, 0);
					//cell0->addNew<Wt::WText>(var->getTemplateName());

					//Wt::log("info") << std::string(typeid(float).name());
				}
			}
			else if (var->getFieldType() == dyno::FieldTypeEnum::State)
			{
				//Wt::log("info") << var->getDescription();
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

void WParameterDataNode::addScalarFieldWidget(Wt::WTable* table, std::string label, dyno::FBase* field, int labelWidth, int widgetWidth)
{
	Wt::WContainerWidget* fw = createFieldWidget(field);
	if (fw)
	{
		std::unique_ptr<Wt::WContainerWidget> mWidget(fw);
		if (fw != nullptr) {
			//this->connect(fw, SIGNAL(fieldChanged()), this, SLOT(contentUpdated()));
			//layout->addWidget(fw, j, 0);
			int row = table->rowCount();
			auto cell0 = table->elementAt(row, 0);
			auto cell1 = table->elementAt(row, 1);

			cell0->addNew<Wt::WText>(label);
			//cell0->setContentAlignment(Wt::AlignmentFlag::Middle);
			//cell0->setWidth(labelWidth);

			//cell1->setContentAlignment(Wt::AlignmentFlag::Middle);
			//cell1->setWidth(widgetWidth);

			cell0->addWidget(std::unique_ptr<Wt::WContainerWidget>(std::move(mWidget)));
		}
	}

}