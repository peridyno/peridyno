#include "PPropertyWidget.h"
#include "Module.h"
#include "Node.h"
#include "FilePath.h"
#include "SceneGraph.h"

#include "NodeEditor/QtNodeWidget.h"
#include "NodeEditor/QtModuleWidget.h"

#include "PCustomWidgets.h"

#include "Common.h"

#include <QGroupBox>
#include <QLabel>
#include <QCheckBox>

#include <QSlider>
#include <QSpinBox>
#include <QRegularExpression>
#include <QMessageBox>


#include <QDoubleValidator>
#include <QPixmap>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QMessageBox>
#include <QFileDialog>
#include <QFile>

namespace dyno
{
	QBoolFieldWidget::QBoolFieldWidget(FBase* field)
		: QGroupBox()
	{

		m_field = field;
		FVar<bool>* f = TypeInfo::cast<FVar<bool>>(m_field);
		if (f == nullptr)
		{
			return;
		}

		this->setStyleSheet("border:none");
		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		name->setFixedSize(100, 18);
		name->setText(FormatFieldWidgetName(field->getObjectName()));
		QCheckBox* checkbox = new QCheckBox();
		//checkbox->setFixedSize(40, 18);
		layout->addWidget(name, 0, 0);
		layout->addWidget(checkbox, 0, 1);

		connect(checkbox, SIGNAL(stateChanged(int)), this, SLOT(changeValue(int)));

		checkbox->setChecked(f->getData());
	}

	void QBoolFieldWidget::changeValue(int status)
	{
		FVar<bool>* f = TypeInfo::cast<FVar<bool>>(m_field);
		if (f == nullptr)
		{
			return;
		}

		if (status == Qt::Checked)
		{
			f->setValue(true);
			f->update();
		}
		else if (status == Qt::PartiallyChecked)
		{
			//m_pLabel->setText("PartiallyChecked");
		}
		else
		{
			f->setValue(false);
			f->update();
		}

		emit fieldChanged();
	}


	QFInstanceWidget::QFInstanceWidget(FBase* field)
		: QGroupBox()
	{


		this->setStyleSheet("border:none");
		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		name->setFixedSize(100, 18);
		name->setText(FormatFieldWidgetName(field->getObjectName()));
		QCheckBox* checkbox = new QCheckBox();
		//checkbox->setFixedSize(40, 18);
		layout->addWidget(name, 0, 0);
		layout->addWidget(checkbox, 0, 1);

		connect(checkbox, SIGNAL(stateChanged(int)), this, SLOT(changeValue2(int)));
	
	}
	void QFInstanceWidget::changeValue(int status)
	{
		printf("QFInstanceWidget check----\n");
		
		emit fieldChanged();
	}
	

	QIntegerFieldWidget::QIntegerFieldWidget(FBase* field)
		: QGroupBox()
	{
		m_field = field;
		FVar<int>* f = TypeInfo::cast<FVar<int>>(m_field);
		if (f == nullptr)
		{
			return;
		}

		this->setStyleSheet("border:none");
		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		name->setFixedSize(100, 18);
		name->setText(FormatFieldWidgetName(field->getObjectName()));

		QSpinBox* spinner = new QSpinBox;
		spinner->setValue(f->getData());

		layout->addWidget(name, 0, 0);
		layout->addWidget(spinner, 0, 1, Qt::AlignRight);


		this->connect(spinner, SIGNAL(valueChanged(int)), this, SLOT(changeValue(int)));
	}

	void QIntegerFieldWidget::changeValue(int value)
	{
		FVar<int>* f = TypeInfo::cast<FVar<int>>(m_field);
		if (f == nullptr)
		{
			return;
		}

		f->setValue(value);
		f->update();

		emit fieldChanged();
	}


	QRealFieldWidget::QRealFieldWidget(FBase* field)
		: QGroupBox()
	{
		m_field = field;
		
		this->setStyleSheet("border:none");
		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		name->setFixedSize(100, 18);
		name->setText(FormatFieldWidgetName(field->getObjectName()));

		QDoubleSlider* slider = new QDoubleSlider;
		slider->setRange(m_field->getMin(), m_field->getMax());
		slider->setMinimumWidth(180);

		QLabel* spc = new QLabel();
		spc->setFixedSize(10, 18);

		QDoubleSpinner* spinner = new QDoubleSpinner;
		spinner->setRange(m_field->getMin(), m_field->getMax());

		layout->addWidget(name, 0, 0);
		layout->addWidget(slider, 0, 1);
		layout->addWidget(spc, 0, 2);
		layout->addWidget(spinner, 0, 3, Qt::AlignRight);

		QObject::connect(slider, SIGNAL(valueChanged(double)), spinner, SLOT(setValue(double)));
		QObject::connect(spinner, SIGNAL(valueChanged(double)), slider, SLOT(setValue(double)));
		QObject::connect(spinner, SIGNAL(valueChanged(double)), this, SLOT(changeValue(double)));

		std::string template_name = field->getTemplateName();
		if (template_name == std::string(typeid(float).name()))
		{
			FVar<float>* f = TypeInfo::cast<FVar<float>>(m_field);
			slider->setValue((double)f->getValue());
		}
		else if(template_name == std::string(typeid(double).name()))
		{
			FVar<double>* f = TypeInfo::cast<FVar<double>>(m_field);
			slider->setValue(f->getValue());
		}

		FormatFieldWidgetName(field->getObjectName());
	}

	void QRealFieldWidget::changeValue(double value)
	{
		std::string template_name = m_field->getTemplateName();

		if (template_name == std::string(typeid(float).name()))
		{
			FVar<float>* f = TypeInfo::cast<FVar<float>>(m_field);
			f->setValue((float)value);
			f->update();
		}
		else if (template_name == std::string(typeid(double).name()))
		{
			FVar<double>* f = TypeInfo::cast<FVar<double>>(m_field);
			f->setValue(value);
			f->update();
		}

		emit fieldChanged();
	}

	mDoubleSpinBox::mDoubleSpinBox(QWidget* parent)
		: QDoubleSpinBox(parent)
	{
		
	}
	void mDoubleSpinBox::wheelEvent(QWheelEvent* event)
	{
		
	}

	QVector3FieldWidget::QVector3FieldWidget(FBase* field)
		: QGroupBox()
	{
		m_field = field;

		this->setStyleSheet("border:none");
		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		name->setFixedSize(100, 18);
		name->setText(FormatFieldWidgetName(field->getObjectName()));

		spinner1 = new mDoubleSpinBox;
		spinner1->setMinimumWidth(30);
		spinner1->setRange(m_field->getMin(), m_field->getMax());
	
		spinner2 = new mDoubleSpinBox;
		spinner2->setMinimumWidth(30);
		spinner2->setRange(m_field->getMin(), m_field->getMax());

		spinner3 = new mDoubleSpinBox;
		spinner3->setMinimumWidth(30);
		spinner3->setRange(m_field->getMin(), m_field->getMax());

		layout->addWidget(name, 0, 0);
		layout->addWidget(spinner1, 0, 1);
		layout->addWidget(spinner2, 0, 2);
		layout->addWidget(spinner3, 0, 3);


		std::string template_name = m_field->getTemplateName();

		double v1 = 0;
		double v2 = 0;
		double v3 = 0;

		if (template_name == std::string(typeid(Vec3f).name()))
		{
			FVar<Vec3f>* f = TypeInfo::cast<FVar<Vec3f>>(m_field);
			auto v = f->getData();
			v1 = v[0];
			v2 = v[1];
			v3 = v[2];
		}
		else if (template_name == std::string(typeid(Vec3d).name()))
		{
			FVar<Vec3d>* f = TypeInfo::cast<FVar<Vec3d>>(m_field);
			auto v = f->getData();

			v1 = v[0];
			v2 = v[1];
			v3 = v[2];
		}

		spinner1->setValue(v1);
		spinner2->setValue(v2);
		spinner3->setValue(v3);

		QObject::connect(spinner1, SIGNAL(valueChanged(double)), this, SLOT(changeValue(double)));
		QObject::connect(spinner2, SIGNAL(valueChanged(double)), this, SLOT(changeValue(double)));
		QObject::connect(spinner3, SIGNAL(valueChanged(double)), this, SLOT(changeValue(double)));
	}


	void QVector3FieldWidget::changeValue(double value)
	{
		double v1 = spinner1->value();
		double v2 = spinner2->value();
		double v3 = spinner3->value();

		std::string template_name = m_field->getTemplateName();

		if (template_name == std::string(typeid(Vec3f).name()))
		{
			FVar<Vec3f>* f = TypeInfo::cast<FVar<Vec3f>>(m_field);
			f->setValue(Vec3f((float)v1, (float)v2, (float)v3));
			f->update();
		}
		else if (template_name == std::string(typeid(Vec3d).name()))
		{
			FVar<Vec3d>* f = TypeInfo::cast<FVar<Vec3d>>(m_field);
			f->setValue(Vec3d(v1, v2, v3));
			f->update();
		}

		emit fieldChanged();
	}

	QStringFieldWidget::QStringFieldWidget(FBase* field) 
		: QGroupBox()
	{
		m_field = field;

		FVar<FilePath>* f = TypeInfo::cast<FVar<FilePath>>(field);

		this->setStyleSheet("border:none");
		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		name->setFixedSize(100, 18);
		name->setText(FormatFieldWidgetName(field->getObjectName()));

		location = new QLineEdit;
		location->setText(QString::fromStdString(f->getValue().string()));

		QPushButton* open = new QPushButton("open");
		open->setStyleSheet("QPushButton{color: black;   border-radius: 10px;  border: 1px groove black;background-color:white; }"
							"QPushButton:hover{background-color:white; color: black;}"  
							"QPushButton:pressed{background-color:rgb(85, 170, 255); border-style: inset; }" );

		layout->addWidget(name, 0, 0);
		layout->addWidget(location, 0, 1);
		layout->addWidget(open, 0, 2);

		connect(location, &QLineEdit::textChanged, this, &QStringFieldWidget::changeValue);

		connect(open, &QPushButton::clicked, this, [=]() {
			QString path = QFileDialog::getOpenFileName(this, tr("Open File"), ".", tr("Text Files(*.*)"));
			if (!path.isEmpty()) {
				QFile file(path);
				if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
					QMessageBox::warning(this, tr("Read File"),
						tr("Cannot open file:\n%1").arg(path));
					return;
				}
				location->setText(path);
				file.close();
			}
			else {
				QMessageBox::warning(this, tr("Path"), tr("You do not select any file."));
			}
		});
	}

	void QStringFieldWidget::changeValue(QString str)
	{
		auto f = TypeInfo::cast<FVar<FilePath>>(m_field);
		if (f == nullptr)
		{
			return;
		}
		f->setValue(str.toStdString());
		f->update();

		emit fieldChanged();
	}

	QStateFieldWidget::QStateFieldWidget(FBase* field)
	{
		m_field = field;

		this->setStyleSheet("border:none");
		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		name->setFixedSize(100, 18);
		name->setText(FormatFieldWidgetName(field->getObjectName()));
		layout->addWidget(name, 0, 0);

		QCheckBox* checkbox = new QCheckBox();		
		layout->addWidget(checkbox, 0, 1);

		if (m_field->parent()->findOutputField(field))
		{
			checkbox->setChecked(true);;
		}
		else
		{
			checkbox->setChecked(false);
		}

		//TODO: use another way
		connect(checkbox, SIGNAL(stateChanged(int)), this, SLOT(tagAsOuput(int)));
	}

	void QStateFieldWidget::tagAsOuput(int status)
	{
		emit stateUpdated(m_field, status);
	}

	//QWidget-->QVBoxLayout-->QScrollArea-->QWidget-->QGridLayout
	PPropertyWidget::PPropertyWidget(QWidget *parent)
		: QWidget(parent)
		, m_main_layout()
	{
		m_main_layout = new QVBoxLayout;
		m_scroll_area = new QScrollArea;

		m_main_layout->setContentsMargins(0, 0, 0, 0);
		m_main_layout->setSpacing(0);
		m_main_layout->setMargin(0);
		m_main_layout->addWidget(m_scroll_area);

		m_scroll_area->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
		m_scroll_area->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
		m_scroll_area->setWidgetResizable(true);

		m_scroll_layout = new QGridLayout;
		m_scroll_layout->setAlignment(Qt::AlignTop);
		m_scroll_layout->setMargin(0);
		
		QWidget * m_scroll_widget = new QWidget;
		m_scroll_widget->setLayout(m_scroll_layout);
		m_scroll_area->setWidget(m_scroll_widget);
		m_scroll_area->setContentsMargins(0, 0, 0, 0);
		//setMinimumWidth(250);
		setLayout(m_main_layout);
	}

	PPropertyWidget::~PPropertyWidget()
	{
		m_widgets.clear();
	}

	QSize PPropertyWidget::sizeHint() const
	{
		return QSize(512, 20);
	}

	QWidget* PPropertyWidget::addWidget(QWidget* widget)
	{
		m_scroll_layout->addWidget(widget);
		m_widgets.push_back(widget);

		return widget;
	}

	void PPropertyWidget::removeAllWidgets()
	{
		//TODO: check whether m_widgets[i] should be explicitly deleted
		for (int i = 0; i < m_widgets.size(); i++)
		{
			m_scroll_layout->removeWidget(m_widgets[i]);
			delete m_widgets[i];
		}
		m_widgets.clear();
	}

	void PPropertyWidget::showProperty(Module* module)
	{
//		clear();

		updateContext(module);

	}

	void PPropertyWidget::showProperty(Node* node)
	{
//		clear();
		updateContext(node);

	}

	void PPropertyWidget::showNodeProperty(Qt::QtNode& block)
	{
		auto dataModel = block.nodeDataModel();

		auto node = dynamic_cast<Qt::QtNodeWidget*>(dataModel);
		if (node != nullptr)
		{
			this->showProperty(node->getNode().get());
		}
		else
		{
			auto module = dynamic_cast<Qt::QtModuleWidget*>(dataModel);
			if (module != nullptr)
			{
				this->showProperty(module->getModule());
			}
		}
	}

	void PPropertyWidget::updateDisplay()
	{
		printf("updateDisplay \n");

		//PVTKOpenGLWidget::getCurrentRenderer()->GetActors()->RemoveAllItems();
		//SceneGraph::getInstance().updateGraphicsContext();
		//PVTKOpenGLWidget::getCurrentRenderer()->GetRenderWindow()->Render();
	}

	void PPropertyWidget::updateContext(OBase* base)
	{
		if (base == nullptr)
		{
			return;
		}

		this->removeAllWidgets();

		QWidget* mWidget = new QWidget;

		std::string mLabel[2] = { {" Control Variables" }, {" State Variables" } };

		int propertyNum[2];

		int n = 2;//label number
		for (int i = 0; i < n; i++) {
			mPropertyLabel[i] = new LockerButton;
			mPropertyLabel[i]->SetTextLabel(QString::fromStdString(mLabel[i]));
			mPropertyLabel[i]->SetImageLabel(QPixmap((getAssetPath() + "/icon/control-270.png").c_str()));
			mPropertyLabel[i]->setStyleSheet("#LockerButton{background-color:transparent}"
				"#LockerButton:hover{background-color:rgba(195,195,195,0.4)}"
				"#LockerButton:pressed{background-color:rgba(127,127,127,0.4)}");

			mPropertyWidget[i] = new QWidget(this);
			mPropertyWidget[i]->setVisible(true);

			propertyNum[i] = 0;

			mPropertyLayout[i] = new QGridLayout;
		}


		std::vector<FBase*>& fields = base->getAllFields();
		for each (FBase * var in fields)
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
					mPropertyLabel[i]->SetImageLabel(QPixmap((getAssetPath() + "/icon/control.png").c_str()));
					//m_sizeList偶数屏蔽Size列表界面，奇数显示Size列表界面
					mPropertyWidget[i]->setVisible(false);
				}
				else
				{
					printf("vlayout->sizeHint().width() - %d \n", vlayout->sizeHint().width());
					mPropertyLabel[i]->SetImageLabel(QPixmap((getAssetPath() + "/icon/control-270.png").c_str()));
					mPropertyWidget[i]->setVisible(true);
				}
				mFlag[i] = !mFlag[i];

			});
		}
		vlayout->setMargin(0);
		vlayout->setSpacing(0);
		mWidget->setLayout(vlayout);
		addWidget(mWidget);
	}

	void PPropertyWidget::addScalarFieldWidget(FBase* field, QGridLayout* layout,int j)
	{
		std::string template_name = field->getTemplateName();

		if (template_name == std::string(typeid(bool).name()))
		{
			auto fw = new QBoolFieldWidget(field);
			this->connect(fw, SIGNAL(fieldChanged()), this, SLOT(updateDisplay()));
			
			layout->addWidget(fw,j,0);
		}
		else if (template_name == std::string(typeid(int).name()))
		{
			auto fw = new QIntegerFieldWidget(field);

			this->connect(fw, SIGNAL(fieldChanged()), this, SLOT(updateDisplay()));

			layout->addWidget(fw, j, 0);
//			this->addWidget(new QIntegerFieldWidget(new FVar<int>()));
		}
		else if (template_name == std::string(typeid(float).name()))
		{
			auto fw = new QRealFieldWidget(field);

			layout->addWidget(fw ,j, 0);
		}
		else if (template_name == std::string(typeid(Vec3f).name()))
		{
			auto fw = new QVector3FieldWidget(field);
			layout->addWidget(fw, j, 0);
		}
		else if (template_name == std::string(typeid(FilePath).name()))
		{
			auto fw = new QStringFieldWidget(field);
			layout->addWidget(fw, j, 0);
		}
	}

	void PPropertyWidget::addArrayFieldWidget(FBase* field)
	{
		auto fw = new QStateFieldWidget(field);
		this->addWidget(fw);
	}

	void PPropertyWidget::addInstanceFieldWidget(FBase* field)
	{
		std::string className = field->getClassName();
		if (className == std::string("FInstance")) {
			auto fw = new QFInstanceWidget(field);
			this->connect(fw, SIGNAL(fieldChanged()), this, SLOT(updateDisplay()));
			this->addWidget(fw);
		}
	}

	void PPropertyWidget::addStateFieldWidget(FBase* field)
	{
		auto widget = new QStateFieldWidget(field);
		connect(widget, &QStateFieldWidget::stateUpdated, this, &PPropertyWidget::fieldUpdated);
		mPropertyLayout[1]->addWidget(widget);
	}
}
