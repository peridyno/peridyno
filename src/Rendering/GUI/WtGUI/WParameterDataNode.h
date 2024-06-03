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
#include "WtGUI/PropertyItem/WVector3FieldWidget.h"
#include "WtGUI/PropertyItem/WVector3iFieldWidget.h"
#include "WtGUI/PropertyItem/WBoolFieldWidget.h"

namespace dyno
{
	class Node;
	class Module;
	class SceneGraph;
	class FBase;
};

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

public:
	struct FieldWidgetMeta {
		using constructor_t = Wt::WContainerWidget* (*)(dyno::FBase*);
		const std::type_info* type;
		constructor_t constructor;
	};

	static int registerWidget(const FieldWidgetMeta& meta);

	static FieldWidgetMeta* getRegistedWidget(const std::string&);

	Wt::WContainerWidget* createFieldWidget(dyno::FBase* field);

private:

	std::shared_ptr<dyno::Node> mNode;
	std::shared_ptr<dyno::Module> mModule;
	Wt::WTable* table;

	//std::unique_ptr<Wt::WContainerWidget> mWidget;

	void addScalarFieldWidget(Wt::WTable* table, std::string label, dyno::FBase* field, int labelWidth = 150, int widgetWidth = 300);

	static std::map<std::string, FieldWidgetMeta> sFieldWidgetMeta;
};

#define DECLARE_FIELD_WIDGET \
	static int reg_field_widget; \
	static Wt::WContainerWidget* createWidget(dyno::FBase*);

#define IMPL_FIELD_WIDGET(_data_type_, _type_) \
	int _type_::reg_field_widget = \
		dyno::WParameterDataNode::registerWidget(dyno::WParameterDataNode::FieldWidgetMeta {&typeid(_data_type_), &_type_::createWidget}); \
	Wt::WContainerWidget* _type_::createWidget(dyno::FBase* f) { return new _type_(f); }