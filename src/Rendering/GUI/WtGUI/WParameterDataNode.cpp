#include "WParameterDataNode.h"
#include <SceneGraph.h>

template<class T>
T* addTableRow(Wt::WTable* table, std::string label, int labelWidth = 200, int widgetWidth = 150)
{
	int row = table->rowCount();
	auto cell0 = table->elementAt(row, 0);
	auto cell1 = table->elementAt(row, 1);

	cell0->addNew<Wt::WText>(label);
	cell0->setContentAlignment(Wt::AlignmentFlag::Middle);
	cell0->setWidth(labelWidth);

	cell1->setContentAlignment(Wt::AlignmentFlag::Middle);
	cell1->setWidth(widgetWidth);

	T* widget = cell1->addNew<T>();
	widget->setWidth(widgetWidth);
	return widget;
}

WParameterDataNode::WParameterDataNode()
{
	//sWContainerWidget.insert(std::pair< std::string, Wt::WContainerWidget>("float", WRealWidget));
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

		std::vector<dyno::FBase*>& fields = mNode->getAllFields();
		for (dyno::FBase* var : fields)
		{
			if (var != nullptr)
			{
				if (var->getFieldType() == dyno::FieldTypeEnum::Param)
				{
					if (var->getClassName() == std::string("FVar"))
					{
						Wt::log("info") << var->getDescription();
					}
				}
				else if (var->getFieldType() == dyno::FieldTypeEnum::State)
				{
					Wt::log("info") << var->getDescription();
				}
			}
		}

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
					Wt::WDoubleSpinBox* test;
					test = addTableRow<Wt::WDoubleSpinBox>(table, var->getTemplateName());
					Wt::log("info") << var->getDescription();
				}
			}
			else if (var->getFieldType() == dyno::FieldTypeEnum::State)
			{
				//Wt::log("info") << var->getDescription();
			}
		}
	}
	//table->elementAt(a, 0)->addWidget(std::make_unique<Wt::WText>("Item @ row 0, column 0"));
	table->setMargin(10);
}