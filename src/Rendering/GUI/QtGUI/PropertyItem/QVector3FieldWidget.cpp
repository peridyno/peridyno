#include "QVector3FieldWidget.h"

#include <QVBoxLayout>

#include "Field.h"
#include "QmDoubleSpinBox.h"


namespace dyno
{
	IMPL_FIELD_WIDGET(Vec3f, QVector3FieldWidget)

	QVector3FieldWidget::QVector3FieldWidget(FBase* field)
		: QFieldWidget(field)
	{
		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		toggleLabel* name = new toggleLabel();
		QString str = FormatFieldWidgetName(field->getObjectName());
		name->setFixedSize(100, 18);
		QFontMetrics fontMetrics(name->font());
		QString elide = fontMetrics.elidedText(str, Qt::ElideRight, 100);
		name->setText(elide);
		//Set label tips
		name->setToolTip(str);

		spinner1 = new mDoubleSpinBox;
		spinner1->setMinimumWidth(30);
		spinner1->setRange(field->getMin(), field->getMax());

		spinner2 = new mDoubleSpinBox;
		spinner2->setMinimumWidth(30);
		spinner2->setRange(field->getMin(), field->getMax());

		spinner3 = new mDoubleSpinBox;
		spinner3->setMinimumWidth(30);
		spinner3->setRange(field->getMin(), field->getMax());

		layout->addWidget(name, 0, 0);
		layout->addWidget(spinner1, 0, 1);
		layout->addWidget(spinner2, 0, 2);
		layout->addWidget(spinner3, 0, 3);
		layout->setSpacing(3);

		std::string template_name = field->getTemplateName();

		double v1 = 0;
		double v2 = 0;
		double v3 = 0;

		if (template_name == std::string(typeid(Vec3f).name()))
		{
			FVar<Vec3f>* f = TypeInfo::cast<FVar<Vec3f>>(field);
			auto v = f->getData();
			v1 = v[0];
			v2 = v[1];
			v3 = v[2];
		}
		else if (template_name == std::string(typeid(Vec3d).name()))
		{
			FVar<Vec3d>* f = TypeInfo::cast<FVar<Vec3d>>(field);
			auto v = f->getData();

			v1 = v[0];
			v2 = v[1];
			v3 = v[2];
		}
		spinner1->ModifyValueAndUpdate(v1);
		spinner2->ModifyValueAndUpdate(v2);
		spinner3->ModifyValueAndUpdate(v3);

		QObject::connect(spinner1, SIGNAL(valueChanged(double)), this, SLOT(updateField(double)));
		QObject::connect(spinner2, SIGNAL(valueChanged(double)), this, SLOT(updateField(double)));
		QObject::connect(spinner3, SIGNAL(valueChanged(double)), this, SLOT(updateField(double)));
		
		QObject::connect(name, SIGNAL(toggle(bool)), spinner1, SLOT(toggleDecimals(bool)));
		QObject::connect(name, SIGNAL(toggle(bool)), spinner2, SLOT(toggleDecimals(bool)));
		QObject::connect(name, SIGNAL(toggle(bool)), spinner3, SLOT(toggleDecimals(bool)));



		QObject::connect(this, SIGNAL(fieldChanged()), this, SLOT(updateWidget()));
	}


	QVector3FieldWidget::QVector3FieldWidget(QString name, Vec3f v)
	{
		value = v;

		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		toggleLabel* nameLabel = new toggleLabel();
		
		nameLabel->setFixedSize(100, 18);
		QFontMetrics fontMetrics(nameLabel->font());
		QString elide = fontMetrics.elidedText(name, Qt::ElideRight, 100);
		nameLabel->setText(elide);
		//Set label tips
		nameLabel->setToolTip(name);

		spinner1 = new mDoubleSpinBox;
		spinner1->setMinimumWidth(30);
		spinner1->setRange(-100000, 100000);

		spinner2 = new mDoubleSpinBox;
		spinner2->setMinimumWidth(30);
		spinner2->setRange(-100000, 100000);

		spinner3 = new mDoubleSpinBox;
		spinner3->setMinimumWidth(30);
		spinner3->setRange(-100000, 100000);


		layout->addWidget(nameLabel, 0, 0);
		layout->addWidget(spinner1, 0, 1);
		layout->addWidget(spinner2, 0, 2);
		layout->addWidget(spinner3, 0, 3);
		layout->setSpacing(3);

		double v1 = value[0];
		double v2 = value[1];
		double v3 = value[2];

		spinner1->setRealValue(v1);
		spinner2->setRealValue(v2);
		spinner3->setRealValue(v3);

		QObject::connect(nameLabel, SIGNAL(toggle(bool)), spinner1, SLOT(toggleDecimals(bool)));
		QObject::connect(nameLabel, SIGNAL(toggle(bool)), spinner2, SLOT(toggleDecimals(bool)));
		QObject::connect(nameLabel, SIGNAL(toggle(bool)), spinner3, SLOT(toggleDecimals(bool)));

		QObject::connect(spinner1, SIGNAL(valueChanged(double)), this, SLOT(vec3fValueChange(double)));
		QObject::connect(spinner2, SIGNAL(valueChanged(double)), this, SLOT(vec3fValueChange(double)));
		QObject::connect(spinner3, SIGNAL(valueChanged(double)), this, SLOT(vec3fValueChange(double)));

		//QObject::connect(this, SIGNAL(fieldChanged()), this, SLOT(updateWidget()));
	}


	QVector3FieldWidget::~QVector3FieldWidget()
	{
	}

	void QVector3FieldWidget::updateField(double value)
	{
		double v1 = spinner1->getRealValue();
		double v2 = spinner2->getRealValue();
		double v3 = spinner3->getRealValue();

		std::string template_name = field()->getTemplateName();

		if (template_name == std::string(typeid(Vec3f).name()))
		{
			FVar<Vec3f>* f = TypeInfo::cast<FVar<Vec3f>>(field());
			f->setValue(Vec3f((float)v1, (float)v2, (float)v3));
		}
		else if (template_name == std::string(typeid(Vec3d).name()))
		{
			FVar<Vec3d>* f = TypeInfo::cast<FVar<Vec3d>>(field());
			f->setValue(Vec3d(v1, v2, v3));
		}
	}


	void QVector3FieldWidget::updateWidget()
	{
		std::string template_name = field()->getTemplateName();

		double v1 = 0;
		double v2 = 0;
		double v3 = 0;

		if (template_name == std::string(typeid(Vec3f).name()))
		{
			FVar<Vec3f>* f = TypeInfo::cast<FVar<Vec3f>>(field());
			auto v = f->getData();
			v1 = v[0];
			v2 = v[1];
			v3 = v[2];
		}
		else if (template_name == std::string(typeid(Vec3d).name()))
		{
			FVar<Vec3d>* f = TypeInfo::cast<FVar<Vec3d>>(field());
			auto v = f->getData();

			v1 = v[0];
			v2 = v[1];
			v3 = v[2];
		}

		spinner1->blockSignals(true);
		spinner2->blockSignals(true);
		spinner3->blockSignals(true);

		spinner1->ModifyValueAndUpdate(v1);
		spinner2->ModifyValueAndUpdate(v2);
		spinner3->ModifyValueAndUpdate(v3);

		spinner1->blockSignals(false);
		spinner2->blockSignals(false);
		spinner3->blockSignals(false);
	}

	void QVector3FieldWidget::vec3fValueChange(double)
	{
		value = Vec3f(spinner1->getRealValue(), spinner2->getRealValue(), spinner3->getRealValue());
		emit vec3fChange(double(value[0]), double(value[1]), double(value[2]));
	}

}

