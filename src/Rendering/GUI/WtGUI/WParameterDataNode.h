#pragma once

#include <Wt/WAbstractItemModel.h>
#include <Wt/WAbstractTableModel.h>
#include <Wt/WText.h>
#include <Wt/WPanel.h>
#include <Wt/WTable.h>
#include <Wt/WDoubleSpinBox.h>
#include <Wt/WLogger.h>

#include <FBase.h>

#include "WtGUI/PropertyItem/WRealFieldWidget.h"

namespace dyno
{
	class Node;
	class Module;
	class SceneGraph;
	class FBase;
};

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

class WParameterDataNode : public Wt::WAbstractTableModel
{
public:

	WParameterDataNode();
	~WParameterDataNode();

	void setNode(std::shared_ptr<dyno::Node> node);
	void setModule(std::shared_ptr<dyno::Module> module);

	virtual int columnCount(const Wt::WModelIndex& parent = Wt::WModelIndex()) const;
	virtual int rowCount(const Wt::WModelIndex& parent = Wt::WModelIndex()) const;

	virtual Wt::cpp17::any data(const Wt::WModelIndex& index,
		Wt::ItemDataRole role = Wt::ItemDataRole::Display) const;

	virtual Wt::cpp17::any headerData(int section,
		Wt::Orientation orientation = Wt::Orientation::Horizontal,
		Wt::ItemDataRole role = Wt::ItemDataRole::Display) const;

	void createParameterPanel(Wt::WPanel* panel);

	void updateNode();
	void updateModule();

	static int registerWidget(const Wt::WContainerWidget);

	static Wt::WContainerWidget* getRegistedWidget(const std::string&);

	//std::shared_ptr<dyno::Module> getModuleField(const Wt::WModelIndex& index);
	//std::shared_ptr<dyno::Node> getNodeField(const Wt::WModelIndex& index);

	struct FieldWidgetMeta {
		using constructor_t = Wt::WContainerWidget* (*)(dyno::FBase*);
		const std::type_info* type;
		constructor_t constructor;
	};
private:

	std::shared_ptr<dyno::Node> mNode;
	std::shared_ptr<dyno::Module> mModule;

	Wt::WTable* table;

	void addScalarFieldWidget(dyno::FBase* field);

	//static std::map < std::string, Wt::WContainerWidget> sWContainerWidget;
};