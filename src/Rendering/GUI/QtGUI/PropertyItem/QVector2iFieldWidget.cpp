#include "QVector2iFieldWidget.h"

#include <QGridLayout>

#include "Field.h"
#include "QPiecewiseSpinBox.h"

namespace dyno
{
	//IMPL_FIELD_WIDGET(Vec2i, QVector2iFieldWidget)
	int QVector2iFieldWidget::reg_field_widget = []() 
	{
		dyno::PPropertyWidget::registerWidget({ &typeid(Vec2i), &QVector2iFieldWidget::createWidget });
		dyno::PPropertyWidget::registerWidget({ &typeid(Vec2u), &QVector2iFieldWidget::createWidget });
		return 0;
	}();

	QWidget* QVector2iFieldWidget::createWidget(dyno::FBase* f) {
		return new QVector2iFieldWidget(f);
	}

	QVector2iFieldWidget::QVector2iFieldWidget(FBase* field)
		: QFieldWidget(field)
	{
		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		//Label
		QLabel* name = new QLabel();
		QString str = FormatFieldWidgetName(field->getObjectName());
		name->setFixedSize(100, 18);
		QFontMetrics fontMetrics(name->font());
		QString elide = fontMetrics.elidedText(str, Qt::ElideRight, 100);
		name->setText(elide);
		//Set label tips
		name->setToolTip(str);

		spinner1 = new QPiecewiseSpinBox;
		spinner1->setMinimumWidth(30);
		spinner1->setRange(castMinimum<int>(field->getMin()), castMaximum<int>(field->getMax()));

		spinner2 = new QPiecewiseSpinBox;
		spinner2->setMinimumWidth(30);
		spinner2->setRange(castMinimum<int>(field->getMin()), castMaximum<int>(field->getMax()));

		layout->addWidget(name, 0, 0);
		layout->addWidget(spinner1, 0, 1);
		layout->addWidget(spinner2, 0, 2);
		layout->setSpacing(3);

		std::string template_name = field->getTemplateName();
		int v1 = 0;
		int v2 = 0;

		if (template_name == std::string(typeid(Vec2i).name()))
		{
			FVar<Vec2i>* f = TypeInfo::cast<FVar<Vec2i>>(field);
			auto v = f->getValue();
			v1 = v[0];
			v2 = v[1];
		}
		else if (template_name == std::string(typeid(Vec2u).name()))
		{
			FVar<Vec2u>* f = TypeInfo::cast<FVar<Vec2u>>(field);
			auto v = f->getValue();
			v1 = v[0];
			v2 = v[1];
			spinner1->setRange(castMinimum<int>(field->getMin()) >= 0 ? castMinimum<int>(field->getMin()) : 0,
				castMaximum<int>(field->getMax()) >=0 ? castMaximum<int>(field->getMax()) : 0);

			spinner2->setRange(castMinimum<int>(field->getMin()) >= 0 ? castMinimum<int>(field->getMin()) : 0,
				castMaximum<int>(field->getMax()) >= 0 ? castMaximum<int>(field->getMax()) : 0);
		}

		spinner1->setValue(v1);
		spinner2->setValue(v2);

		QObject::connect(spinner1, SIGNAL(valueChanged(int)), this, SLOT(updateField(int)));
		QObject::connect(spinner2, SIGNAL(valueChanged(int)), this, SLOT(updateField(int)));

		QObject::connect(this, SIGNAL(fieldChanged()), this, SLOT(updateWidget()));
	}

	QVector2iFieldWidget::~QVector2iFieldWidget()
	{
	}

	void QVector2iFieldWidget::updateField(int)
	{
		int v1 = spinner1->value();
		int v2 = spinner2->value();

		std::string template_name = field()->getTemplateName();

		if (template_name == std::string(typeid(Vec2i).name()))
		{
			FVar<Vec2i>* f = TypeInfo::cast<FVar<Vec2i>>(field());
			f->setValue(Vec2i(v1, v2),false);
		}
		else if (template_name == std::string(typeid(Vec2u).name()))
		{
			FVar<Vec2u>* f = TypeInfo::cast<FVar<Vec2u>>(field());
			f->setValue(Vec2u(v1, v2),false);
		}

		emit fieldChanged();
	}

	void QVector2iFieldWidget::updateWidget()
	{
		std::string template_name = field()->getTemplateName();

		int v1 = 0;
		int v2 = 0;

		if (template_name == std::string(typeid(Vec2i).name()))
		{
			FVar<Vec2i>* f = TypeInfo::cast<FVar<Vec2i>>(field());
			auto v = f->getValue();
			v1 = v[0];
			v2 = v[1];
		}
		else if (template_name == std::string(typeid(Vec2u).name()))
		{
			FVar<Vec2u>* f = TypeInfo::cast<FVar<Vec2u>>(field());
			auto v = f->getValue();
			v1 = v[0];
			v2 = v[1];
		}

		spinner1->blockSignals(true);
		spinner2->blockSignals(true);

		spinner1->setValue(v1);
		spinner2->setValue(v2);

		spinner1->blockSignals(false);
		spinner2->blockSignals(false);
	}
}
