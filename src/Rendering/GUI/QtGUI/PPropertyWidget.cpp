#include "PPropertyWidget.h"

//Framework
#include "Module.h"
#include "Node.h"
#include "FilePath.h"
#include "Ramp.h"
#include "SceneGraph.h"

//Node editor
#include "NodeEditor/QtNodeWidget.h"
#include "NodeEditor/QtModuleWidget.h"

#include "LockerButton.h"
#include <QVBoxLayout>
#include <QScrollArea>
#include <QGridLayout>

#include "Color.h"

#include "PropertyItem/QVector3iFieldWidget.h"
#include "PropertyItem/QVector3FieldWidget.h"
#include "PropertyItem/QBoolFieldWidget.h"
#include "PropertyItem/QIntegerFieldWidget.h"
#include "PropertyItem/QFilePathWidget.h"
#include "PropertyItem/QRealFieldWidget.h"
#include "PropertyItem/QEnumFieldWidget.h"
#include "PropertyItem/QRampWidget.h"
#include "PropertyItem/QStateFieldWidget.h"
#include "PropertyItem/QColorWidget.h"

namespace dyno
{
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
			mPropertyLabel[i]->SetImageLabel(QPixmap((getAssetPath() + "/icon/arrow_down_pressed.png").c_str()));
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
				if (var->getFieldType() == FieldTypeEnum::Param)
				{
					if (var->getClassName() == std::string("FVar"))
					{
						this->addScalarFieldWidget(var, mPropertyLayout[0], propertyNum[0]);
						propertyNum[0]++;
					}
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
			else
			{
				vlayout->addWidget(mPropertyLabel[i]);
				mPropertyWidget[i]->setLayout(mPropertyLayout[i]);
				vlayout->addWidget(mPropertyWidget[i]);
				//以下为是否再属性栏显示 StateVariables
				mPropertyWidget[i]->setVisible(false);
				mPropertyLabel[i]->setVisible(false);
			}

			connect(mPropertyLabel[i], &LockerButton::clicked, [this, i, vlayout]() {
				if (!mFlag[i])
				{
					mPropertyLabel[i]->SetImageLabel(QPixmap((getAssetPath() + "/icon/arrow_right_pressed.png").c_str()));
					mPropertyWidget[i]->setVisible(false);
				}
				else
				{
					mPropertyLabel[i]->SetImageLabel(QPixmap((getAssetPath() + "/icon/arrow_down_pressed.png").c_str()));
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
			mPropertyLabel[i]->SetImageLabel(QPixmap((getAssetPath() + "/icon/arrow_down_pressed.png").c_str()));
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
					if (var->getClassName() == std::string("FVar"))
					{
						this->addScalarFieldWidget(var, mPropertyLayout[0], propertyNum[0]);
						propertyNum[0]++;
					}
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
					mPropertyLabel[i]->SetImageLabel(QPixmap((getAssetPath() + "/icon/arrow_right_pressed.png").c_str()));
					mPropertyWidget[i]->setVisible(false);
				}
				else
				{
					mPropertyLabel[i]->SetImageLabel(QPixmap((getAssetPath() + "/icon/arrow_down_pressed.png").c_str()));
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

	void PPropertyWidget::contentUpdated()
	{
		auto node = std::dynamic_pointer_cast<Node>(mSeleted);

		if (node != nullptr)
			emit nodeUpdated(node);

		auto module = std::dynamic_pointer_cast<Module>(mSeleted);

		if (module != nullptr)
			emit moduleUpdated(module);
	}

	void PPropertyWidget::addScalarFieldWidget(FBase* field, QGridLayout* layout,int j)
	{
		std::string template_name = field->getTemplateName();

		if (template_name == std::string(typeid(bool).name()))
		{
			auto fw = new QBoolFieldWidget(field);
			this->connect(fw, SIGNAL(fieldChanged()), this, SLOT(contentUpdated()));
			
			layout->addWidget(fw,j,0);
		}
		else if (template_name == std::string(typeid(int).name()))
		{
			auto fw = new QIntegerFieldWidget(field);

			this->connect(fw, SIGNAL(fieldChanged()), this, SLOT(contentUpdated()));

			layout->addWidget(fw, j, 0);
		}
		else if (template_name == std::string(typeid(uint).name()))
		{
			auto fw = new QUIntegerFieldWidget(field);

			this->connect(fw, SIGNAL(fieldChanged()), this, SLOT(contentUpdated()));

			layout->addWidget(fw, j, 0);
		}
		else if (template_name == std::string(typeid(float).name()))
		{
			auto fw = new QRealFieldWidget(field);
			this->connect(fw, SIGNAL(fieldChanged()), this, SLOT(contentUpdated()));

			layout->addWidget(fw ,j, 0);
		}
		else if (template_name == std::string(typeid(Vec3f).name()))
		{
			auto fw = new QVector3FieldWidget(field);
			this->connect(fw, SIGNAL(fieldChanged()), this, SLOT(contentUpdated()));

			layout->addWidget(fw, j, 0);
		}
		else if (template_name == std::string(typeid(Vec3i).name()))
		{
			auto fw = new QVector3iFieldWidget(field);
			this->connect(fw, SIGNAL(fieldChanged()), this, SLOT(contentUpdated()));

			layout->addWidget(fw, j, 0);
		}
		else if (template_name == std::string(typeid(FilePath).name()))
		{
			auto fw = new QFilePathWidget(field);
			this->connect(fw, SIGNAL(fieldChanged()), this, SLOT(contentUpdated()));

			layout->addWidget(fw, j, 0);
		}
		else if (template_name == std::string(typeid(Ramp).name()))
		{
			auto fw = new QRampWidget(field);
			this->connect(fw, SIGNAL(fieldChanged()), this, SLOT(contentUpdated()));

			layout->addWidget(fw, j, 0);
		}
		else if (template_name == std::string(typeid(PEnum).name()))
		{
			auto fw = new QEnumFieldWidget(field);
			//this->connect(fw, SIGNAL(fieldChanged()), this, SLOT(contentUpdated()));

			layout->addWidget(fw, j, 0);
		}
		else if (template_name == std::string(typeid(std::string).name()))
		{
			auto fw = new QStringFieldWidget(field);
			this->connect(fw, SIGNAL(fieldChanged()), this, SLOT(contentUpdated()));

			layout->addWidget(fw, j, 0);
		}
		else if (template_name == std::string(typeid(Color).name()))
		{
			auto fw = new QColorWidget(field);
			this->connect(fw, SIGNAL(fieldChanged()), this, SLOT(contentUpdated()));

			layout->addWidget(fw, j, 0);
		}
	}

	void PPropertyWidget::addArrayFieldWidget(FBase* field)
	{
		auto fw = new QStateFieldWidget(field);
		this->addWidget(fw);
	}

	void PPropertyWidget::addStateFieldWidget(FBase* field)
	{
		auto widget = new QStateFieldWidget(field);
		connect(widget, &QStateFieldWidget::stateUpdated, this, &PPropertyWidget::stateFieldUpdated);
		mPropertyLayout[1]->addWidget(widget);
	}
}
