#include "PPropertyWidget.h"

//Framework
#include "Module.h"
#include "Node.h"
#include "Tuple.h"
#include "Field/FList.h"
#include "SceneGraph.h"

//Node editor
#include "NodeEditor/QtNodeWidget.h"
#include "NodeEditor/QtModuleWidget.h"
#include "nodes/QNode"

#include "LockerButton.h"
#include <QVBoxLayout>
#include <QScrollArea>
#include <QGridLayout>
#include <QRadioButton>
#include <QToolButton>

#include "PropertyItem/QStateFieldWidget.h"
#include "PropertyItem/QArrayWidget.h"
#include "PropertyItem/DescriptionHelper.h"

#include "PropertyItem/QEnumFieldWidget.h"

namespace dyno
{
	std::map<std::string, PPropertyWidget::FieldWidgetMeta> PPropertyWidget::sFieldWidgetMeta {};

	//QWidget-->QVBoxLayout-->QScrollArea-->QWidget-->QGridLayout
	PPropertyWidget::PPropertyWidget(QWidget *parent)
		: QWidget(parent)
		, mMainLayout()
	{
		mMainLayout = new QVBoxLayout;
		mScrollArea = new QScrollArea;

		mMainLayout->setContentsMargins(0, 0, 0, 0);
		mMainLayout->setSpacing(0);
		mMainLayout->addWidget(mScrollArea);

		mScrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
		mScrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
		mScrollArea->setWidgetResizable(true);

		mScrollLayout = new QGridLayout;
		mScrollLayout->setAlignment(Qt::AlignTop);
		mScrollLayout->setContentsMargins(0, 0, 0, 0);
		
		QWidget * m_scroll_widget = new QWidget;
		m_scroll_widget->setLayout(mScrollLayout);
		mScrollArea->setWidget(m_scroll_widget);
		mScrollArea->setContentsMargins(0, 0, 0, 0);
		//setMinimumWidth(250);
		setLayout(mMainLayout);
	}

	PPropertyWidget::~PPropertyWidget()
	{
		mPropertyItems.clear();
	}

	int PPropertyWidget::registerWidget(const FieldWidgetMeta& meta) {
		sFieldWidgetMeta[meta.type->name()] = meta;
		return 0;
	}
	PPropertyWidget::FieldWidgetMeta* PPropertyWidget::getRegistedWidget(const std::string& name) {
		if(sFieldWidgetMeta.count(name))
			return &sFieldWidgetMeta.at(name);
		return nullptr;
	}

	QSize PPropertyWidget::sizeHint() const
	{
		return QSize(512, 20);
	}

	QWidget* PPropertyWidget::addWidget(QWidget* widget)
	{
		mScrollLayout->addWidget(widget);
		mPropertyItems.push_back(widget);

		return widget;
	}

	void PPropertyWidget::removeAllWidgets()
	{
		//TODO: check whether m_widgets[i] should be explicitly deleted
		for (int i = 0; i < mPropertyItems.size(); i++)
		{
			mScrollLayout->removeWidget(mPropertyItems[i]);
			delete mPropertyItems[i];
		}
		mPropertyItems.clear();
	}

	void PPropertyWidget::showModuleProperty(std::shared_ptr<Module> module)
	{
		if (module == nullptr)
			return;

		this->removeAllWidgets();

		QWidget* mWidget = new QWidget;

		std::string mLabel[2] = { {" Control Variables" }, {" State Variables" } };

		int propertyNum[2];

		int n = 2;//label number
		for (int i = 0; i < n; i++) {
			mPropertyLabel[i] = new LockerButton;
			mPropertyLabel[i]->setContentsMargins(8, 0, 0, 0);
			mPropertyLabel[i]->SetTextLabel(QString::fromStdString(mLabel[i]));
			mPropertyLabel[i]->SetImageLabel(QPixmap((getDefaultPath() + "/icon/arrow_down_pressed.png").c_str()));
			mPropertyLabel[i]->GetTextHandle()->setAttribute(Qt::WA_TransparentForMouseEvents, true);
			mPropertyLabel[i]->GetImageHandle()->setAttribute(Qt::WA_TransparentForMouseEvents, true);

			mPropertyWidget[i] = new QWidget(this);
			mPropertyWidget[i]->setVisible(true);
			mPropertyWidget[i]->setStyleSheet("background-color: transparent;");//"color:red;background-color:green;"

			propertyNum[i] = 0;

			mPropertyLayout[i] = new QGridLayout;
		}

		std::vector<FBase*>& fields = module->getAllFields();
		for  (FBase * var : fields)
		{
			if (var != nullptr) {
				this->addVariableFieldWidget(var, mPropertyLayout[0]);
				propertyNum[0]++;
			}
		}

		QVBoxLayout* vlayout = new QVBoxLayout;

		for (int i = 0; i < n; i++) {
			mFlag[i] = false;

			if (propertyNum[i] != 0) {
				vlayout->addWidget(mPropertyLabel[i]);
				mPropertyWidget[i]->setLayout(mPropertyLayout[i]);
				vlayout->addWidget(mPropertyWidget[i]);
			}
			else
			{
				vlayout->addWidget(mPropertyLabel[i]);
				mPropertyWidget[i]->setLayout(mPropertyLayout[i]);
				vlayout->addWidget(mPropertyWidget[i]);
				
				mPropertyWidget[i]->setVisible(false);
				mPropertyLabel[i]->setVisible(false);
			}

			connect(mPropertyLabel[i], &LockerButton::clicked, [this, i, vlayout]() {
				if (!mFlag[i])
				{
					mPropertyLabel[i]->SetImageLabel(QPixmap((getDefaultPath() + "/icon/arrow_right_pressed.png").c_str()));
					mPropertyWidget[i]->setVisible(false);
				}
				else
				{
					mPropertyLabel[i]->SetImageLabel(QPixmap((getDefaultPath() + "/icon/arrow_down_pressed.png").c_str()));
					mPropertyWidget[i]->setVisible(true);
				}
				mFlag[i] = !mFlag[i];
				});
		}
		vlayout->setContentsMargins(0, 0, 0, 0);
		vlayout->setSpacing(0);
		mWidget->setLayout(vlayout);

		addWidget(mWidget);

		mSeleted = module;
	}

	void PPropertyWidget::showNodeProperty(std::shared_ptr<Node> node)
	{
		if (node == nullptr)
			return;

		this->removeAllWidgets();

		QWidget* mWidget = new QWidget;

		std::string mLabel[2] = { {" Control Variables" }, {" State Variables" } };

		int propertyNum[2];

		int n = 2;//label number
		for (int i = 0; i < n; i++) {
			mPropertyLabel[i] = new LockerButton;
			mPropertyLabel[i]->setContentsMargins(8, 0, 0, 0);
			mPropertyLabel[i]->SetTextLabel(QString::fromStdString(mLabel[i]));
			mPropertyLabel[i]->SetImageLabel(QPixmap((getDefaultPath() + "/icon/arrow_down_pressed.png").c_str()));
			mPropertyLabel[i]->GetTextHandle()->setAttribute(Qt::WA_TransparentForMouseEvents, true);
			mPropertyLabel[i]->GetImageHandle()->setAttribute(Qt::WA_TransparentForMouseEvents, true);

			mPropertyWidget[i] = new QWidget(this);
			mPropertyWidget[i]->setVisible(true);
			mPropertyWidget[i]->setStyleSheet("background-color: transparent;");


			propertyNum[i] = 0;

			mPropertyLayout[i] = new QGridLayout;
		}


		{
			QGroupBox* title = new QGroupBox;
			//title->setStyleSheet("border:none");
			QGridLayout* layout = new QGridLayout;
			layout->setContentsMargins(0, 0, 0, 0);
			layout->setSpacing(0);

			title->setLayout(layout);

			QLabel* name = new QLabel();
			//name->setStyleSheet("font: bold; background-color: rgb(230, 230, 230);");
			name->setFixedHeight(25);
			name->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
			name->setText("Name");
			layout->addWidget(name, 0, 0);

			QLabel* output = new QLabel();
			//output->setStyleSheet("font: bold; background-color: rgb(230, 230, 230);");
			output->setFixedSize(64, 25);
			output->setText("Output");
			layout->addWidget(output, 0, 1, Qt::AlignRight);

			mPropertyLayout[1]->addWidget(title); 
		}

		std::vector<FBase*>& fields = node->getAllFields();
		for  (FBase * var : fields)
		{
			if (var != nullptr) {
				if (var->getFieldType() == FieldTypeEnum::Param)
				{
					this->addVariableFieldWidget(var, mPropertyLayout[0]);
					propertyNum[0]++;
				}
				else if (var->getFieldType() == FieldTypeEnum::State) {
					this->addStateFieldWidget(var);
					propertyNum[1]++;
				}
			}
		}

		QVBoxLayout* vlayout = new QVBoxLayout;

		for (int i = 0; i < n; i++) {
			mFlag[i] = false;

			if (propertyNum[i] != 0) {
				vlayout->addWidget(mPropertyLabel[i]);
				mPropertyWidget[i]->setLayout(mPropertyLayout[i]);
				vlayout->addWidget(mPropertyWidget[i]);
			}

			connect(mPropertyLabel[i], &LockerButton::clicked, [this, i, vlayout]() {
				if (!mFlag[i])
				{
					mPropertyLabel[i]->SetImageLabel(QPixmap((getDefaultPath() + "/icon/arrow_right_pressed.png").c_str()));
					mPropertyWidget[i]->setVisible(false);
				}
				else
				{
					mPropertyLabel[i]->SetImageLabel(QPixmap((getDefaultPath() + "/icon/arrow_down_pressed.png").c_str()));
					mPropertyWidget[i]->setVisible(true);
				}
				mFlag[i] = !mFlag[i];
			});
		}
		vlayout->setContentsMargins(0, 0, 0, 0);
		vlayout->setSpacing(0);
		mWidget->setLayout(vlayout);

		addWidget(mWidget);

		mSeleted = node;
	}

	LockerButton* PPropertyWidget::getLockerButton(uint i) 
	{ 
		return mPropertyLabel[i]; 
	};

	void PPropertyWidget::showProperty(Qt::QtNode& block)
	{
		auto dataModel = block.nodeDataModel();

		auto node = dynamic_cast<Qt::QtNodeWidget*>(dataModel);
		if (node != nullptr)
		{
			this->showNodeProperty(node->getNode());
		}
		else
		{
			auto module = dynamic_cast<Qt::QtModuleWidget*>(dataModel);
			if (module != nullptr)
			{
				this->showModuleProperty(module->getModule());
			}
		}
	}

	void PPropertyWidget::clearProperty()
	{
		this->removeAllWidgets();
	}

	void PPropertyWidget::reBuildItem()
	{
		clearProperty();

		if (mSeleted) 
		{
			std::shared_ptr<Node> nodePtr = std::dynamic_pointer_cast<Node>(mSeleted);
			if (nodePtr)
			{
				showNodeProperty(nodePtr);
			}

			std::shared_ptr<Module> modulePtr = std::dynamic_pointer_cast<Module>(mSeleted);
			if (modulePtr)
			{
				this->showModuleProperty(modulePtr);
			}
		}
	}

	void PPropertyWidget::contentUpdated()
	{
		auto node = std::dynamic_pointer_cast<Node>(mSeleted);

		//Update the visibility of each item
		for (int i = 0; i < mPropertyLayout[0]->count(); i++)
		{
			auto layoutItem = mPropertyLayout[0]->itemAt(i);
			if (layoutItem)
			{
				QFieldWidget* widget = dynamic_cast<QFieldWidget*>(layoutItem->widget());
				if (widget != nullptr)
				{
					auto field = widget->field();
					widget->setVisible(field->isActive());
				}
			}
		}
		
		if (node != nullptr) 
		{
			emit nodeUpdated(node);
		}

		auto module = std::dynamic_pointer_cast<Module>(mSeleted);

		if (module != nullptr)
			emit moduleUpdated(module);

	}

	QWidget* PPropertyWidget::createFieldWidget(FBase* field) {
		QWidget* fw {nullptr};
		std::string template_name = field->getTemplateName();

		auto reg = getRegistedWidget(template_name);
		if(reg) {
			fw = reg->constructor(field);
		}

		return fw;
	}

	QWidget* PPropertyWidget::addScalarFieldWidget(FBase* field, QGridLayout* layout)
	{
		if (field->getClassName() != std::string("FVar")) return nullptr;
		if (field->isEmpty())
			return NULL;

		QWidget* fw = createFieldWidget(field);
		
		QEnumFieldWidget* enumWidget = qobject_cast<QEnumFieldWidget*>(fw);
		if (enumWidget)
		{
			connect(enumWidget, &QFieldWidget::fieldChanged, this, [this]() {
				QMetaObject::invokeMethod(this, &PPropertyWidget::reBuildItem, Qt::QueuedConnection);
				}, Qt::DirectConnection);
		}

		if(fw != nullptr) {
			this->connect(fw, SIGNAL(fieldChanged()), this, SLOT(contentUpdated()));

			//Hide the item if the field is not active in default
			if (!field->isActive())
				fw->setVisible(false);

			if(layout != nullptr) layout->addWidget(fw);
		}
		else 
		{
			std::cout <<(field->getObjectName());
		}

		return fw;
	}

	QWidget* PPropertyWidget::addTupleFieldWidget(FTuple* tuple, QGridLayout* layout)
	{
		QString name = FormatFieldWidgetName(tuple->getObjectName());
		
		if (!tuple->getObjectName().empty()) {
			char first = tuple->getObjectName()[0];
			if (std::isalpha(static_cast<unsigned char>(first))) {
				name = QString("[ ") + name + QString("]");
			}
		}

		QGroupBox* groupBox = new QGroupBox;
		groupBox->setStyleSheet(R"(
			QGroupBox {
				margin-top: 12px;
				border: 2px solid #454545;
			}
			QGroupBox::title {
				subcontrol-origin: margin;
				subcontrol-position: top center;
			}
		)");

		QVBoxLayout* mainLayout = new QVBoxLayout;

		LockerButton* toggleButton = new LockerButton;
		toggleButton->SetImageLabel(QPixmap((getDefaultPath() + "/icon/arrow_right_pressed.png").c_str()));
		toggleButton->SetTextLabel(QString("[ ") + FormatFieldWidgetName(tuple->getObjectName()) + QString(" Fields") + QString("]"));
		setContentsMargins(8, 0, 0, 0);
		toggleButton->setCheckable(true);
		toggleButton->setChecked(false);
		toggleButton->setStyleSheet(R"(
			LockerButton {
				background-color: #464646;
				border: 1px solid #000000;
				border-radius: 4px;
				padding: 4px 8px;
				text-align: left;
				color: white;
			}
			LockerButton:hover {
				background-color: #616161;
			}
			LockerButton:pressed {
				background-color: #000000;
			}
		)");

		QWidget* contentWidget = new QWidget;
		contentWidget->setVisible(false);
		contentWidget->setStyleSheet("background-color: transparent;");

		QVBoxLayout* contentLayout = new QVBoxLayout(contentWidget);
		contentLayout->setContentsMargins(0, 4, 0, 0);
		contentLayout->setSpacing(4);

		struct GroupInfo {
			std::string name;
			QGroupBox* groupBox;
			LockerButton* button;
			QWidget* content;
			QGridLayout* layout;
			QBoxLayout* currentLayout;
			enum LastLayoutMode { None, VBox, HBox };
			LastLayoutMode lastMode;
			int row;
			int tempRow;
		};

		std::vector<GroupInfo> groups;
		std::string dummy;

		auto findOrCreateGroup = [&](const std::string& groupName) -> GroupInfo* {
			if (groupName.empty()) {
				return nullptr;
			}
			for (auto& g : groups) {
				if (g.name == groupName) {
					return &g;
				}
			}
			groups.push_back({});
			GroupInfo* g = &groups.back();
			g->name = groupName;
			g->lastMode = GroupInfo::None;
			g->row = -1;
			g->tempRow = -1;

			g->groupBox = new QGroupBox;
			g->groupBox->setStyleSheet(R"(
				QGroupBox {
					margin-top: 10px;
					border: 1px solid #454545;
					border-radius: 4px;
				}
				QGroupBox::title {
					subcontrol-origin: margin;
					subcontrol-position: top center;
				}
			)");

			QVBoxLayout* vbl = new QVBoxLayout(g->groupBox);
			vbl->setContentsMargins(4, 4, 4, 4);

			g->button = new LockerButton;
			g->button->SetImageLabel(QPixmap((getDefaultPath() + "/icon/arrow_right_pressed.png").c_str()));
			g->button->SetTextLabel(QString::fromStdString(groupName));
			g->button->setCheckable(true);
			g->button->setChecked(false);
			g->button->setStyleSheet(R"(
				LockerButton {
					background-color: #464646;
					border: 1px solid #000000;
					border-radius: 4px;
					padding: 3px 8px;
					text-align: left;
					color: white;
				}
				LockerButton:hover {
					background-color: #616161;
				}
				LockerButton:pressed {
					background-color: #000000;
				}
			)");

			g->content = new QWidget;
			g->content->setVisible(false);
			g->content->setStyleSheet("background-color: transparent;");

			g->layout = new QGridLayout(g->content);
			g->layout->setContentsMargins(8, 2, 2, 2);

			vbl->addWidget(g->button);
			vbl->addWidget(g->content);

			LockerButton* btn = g->button;
			QWidget* cw = g->content;
			connect(btn, &LockerButton::clicked, [btn, cw]() {
				if (cw->isVisible()) {
					btn->SetImageLabel(QPixmap((getDefaultPath() + "/icon/arrow_right_pressed.png").c_str()));
					cw->setVisible(false);
				} else {
					btn->SetImageLabel(QPixmap((getDefaultPath() + "/icon/arrow_down_pressed.png").c_str()));
					cw->setVisible(true);
				}
			});

			contentLayout->addWidget(g->groupBox);
			return g;
		};

		QBoxLayout* currentLayout = NULL;
		enum LastLayoutMode
		{
			None,
			VBox,
			HBox
		};

		LastLayoutMode lastMode = None;
		int row = -1;
		int temp = -1;

		QGridLayout* vbox = new QGridLayout;
		bool hasUngrouped = false;

		for (size_t i = 0; i < tuple->size(); i++)
		{
			auto field = tuple->get(i);
			bool selfVLayout = true;
			bool nextVLayout = true;
			bool onlyDetail = false;

			DescriptionHelper::parseQtStyleDescriptionRobust(field->getDescription(), dummy, selfVLayout, onlyDetail);
			std::string groupName = DescriptionHelper::parseQtStyleGroup(field->getDescription());

			if (i < tuple->size() - 1)
				DescriptionHelper::parseQtStyleDescriptionRobust(tuple->get(i + 1)->getDescription(), dummy, nextVLayout, onlyDetail);

			QWidget* fw = this->addVariableFieldWidget(field, NULL);
			if (!fw)
				continue;

			if (!groupName.empty()) {
				GroupInfo* g = findOrCreateGroup(groupName);
				if (g) {
					switch (g->lastMode)
					{
					case GroupInfo::None:
						if (!nextVLayout) { g->currentLayout = new QHBoxLayout; g->lastMode = GroupInfo::HBox; }
						else { g->currentLayout = new QVBoxLayout; g->lastMode = GroupInfo::VBox; }
						g->row++;
						break;
					case GroupInfo::VBox:
						if (!nextVLayout) { g->currentLayout = new QHBoxLayout; g->row++; g->lastMode = GroupInfo::HBox; }
						break;
					case GroupInfo::HBox:
						if (selfVLayout) {
							if (!nextVLayout) { g->currentLayout = new QHBoxLayout; g->lastMode = GroupInfo::HBox; }
							else { g->currentLayout = new QVBoxLayout; g->lastMode = GroupInfo::VBox; }
							g->row++;
						}
						break;
					default:
						break;
					}
					g->currentLayout->addWidget(fw);
					if (g->tempRow != g->row)
						g->layout->addLayout(g->currentLayout, g->row, 0);
					g->tempRow = g->row;
					continue;
				}
			}

			hasUngrouped = true;
			switch (lastMode)
			{
			case None:
				if (!nextVLayout) 
				{
					currentLayout = new QHBoxLayout;
					lastMode = HBox;
				}
				else 
				{
					currentLayout = new QVBoxLayout;
					lastMode = VBox;
				}
				row++;
				break;
			case VBox:
				if (!nextVLayout) 
				{
					currentLayout = new QHBoxLayout;
					row++;
					lastMode = HBox;
				}
				break;
			case HBox:
				if (selfVLayout) 
				{
					if (!nextVLayout)
					{
						currentLayout = new QHBoxLayout;
						lastMode = HBox;
					}
					else
					{
						currentLayout = new QVBoxLayout;
						lastMode = VBox;
					}
					row++;
				}
				break;
			default:
				break;
			}

			currentLayout->addWidget(fw);

			if (temp != row)
				vbox->addLayout(currentLayout,row,0);
			temp = row;
		}

		if (hasUngrouped) {
			contentLayout->insertLayout(0, vbox);
		}

		mainLayout->addWidget(toggleButton);
		mainLayout->addWidget(contentWidget);
		groupBox->setLayout(mainLayout);

		connect(toggleButton, &LockerButton::clicked, [toggleButton, contentWidget]() {
			bool visible = contentWidget->isVisible();
			if (visible)
			{
				toggleButton->SetImageLabel(QPixmap((getDefaultPath() + "/icon/arrow_right_pressed.png").c_str()));
				contentWidget->setVisible(false);
			}
			else
			{
				toggleButton->SetImageLabel(QPixmap((getDefaultPath() + "/icon/arrow_down_pressed.png").c_str()));
				contentWidget->setVisible(true);
			}
		});

		if(layout != nullptr) layout->addWidget(groupBox);

		return groupBox;
	}

	QWidget* PPropertyWidget::addListFieldWidget(FList* field, QGridLayout* layout)
	{
		if (field->getClassName() != std::string("FList")) return nullptr;

		if (auto f = dynamic_cast<FList*> (field))
		{
			auto aw = new ArrayWidget(f);

			this->connect(aw, SIGNAL(fieldChanged()), this, SLOT(contentUpdated()));

			connect(aw, &ArrayWidget::onExpandList, [this, aw]() {
				int id = 0;
				for (auto it = aw->list()->begin(); it != aw->list()->end(); ++it)
				{
					auto childField = (*it).get();
					childField->setObjectName("[" + std::to_string(id) + "]");
					QWidget* fw = addVariableFieldWidget(childField);
					aw->addItem(fw, id, childField);
					id++;
				}			
			});

			connect(aw, &ArrayWidget::onAddElement, [this, aw]()
			{				
				auto it = aw->list()->end();
				--it;
				auto childField = (*it).get();
				QWidget* fw = addVariableFieldWidget(childField);
				aw->addItem(fw, aw->list()->size() - 1, childField);
			});

			if (layout != nullptr) layout->addWidget(aw);

			return aw;
		}

		return nullptr;
	}

	QWidget* PPropertyWidget::addVariableFieldWidget(FBase* field, QGridLayout* layout)
	{
		if (field->getClassName() == std::string("FVar"))
		{
			return this->addScalarFieldWidget(field, layout);
		}
		else if (field->getClassName() == std::string("FTuple"))
		{
			auto tuple = dynamic_cast<FTuple*> (field);
			if (tuple != nullptr)
			{
				return addTupleFieldWidget(tuple, layout);
			}
		}
		else if (field->getClassName() == std::string("FList"))
		{
			auto flist = dynamic_cast<FList*> (field);
			return addListFieldWidget(flist, layout);
		}

		return nullptr;
	}

	void PPropertyWidget::addStateFieldWidget(FBase* field)
	{
		auto widget = new QStateFieldWidget(field);
		connect(widget, &QStateFieldWidget::stateUpdated, this, &PPropertyWidget::stateFieldUpdated);
		mPropertyLayout[1]->addWidget(widget);
	}
}


