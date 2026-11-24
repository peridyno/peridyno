#include "QQuatFieldWidget.h"

#include <QVBoxLayout>

#include "Field.h"
#include "QPiecewiseDoubleSpinBox.h"

namespace dyno
{
	int QQuatFieldWidget::reg_field_widget = []()
	{
		dyno::PPropertyWidget::registerWidget({ &typeid(Quat1f), &QQuatFieldWidget::createWidget });
		dyno::PPropertyWidget::registerWidget({ &typeid(Quat1d), &QQuatFieldWidget::createWidget });
		return 0;
	}();

	QWidget* QQuatFieldWidget::createWidget(dyno::FBase* f) {
		return new QQuatFieldWidget(f);
	}

	QQuatFieldWidget::QQuatFieldWidget(FBase* field)
		: QFieldWidget(field)
	{
		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QToggleLabel* name = new QToggleLabel();
		QString str = FormatFieldWidgetName(field->getObjectName());
		name->setFixedSize(100, 18);
		QFontMetrics fontMetrics(name->font());
		QString elide = fontMetrics.elidedText(str, Qt::ElideRight, 100);
		name->setText(elide);
		name->setToolTip(str);

		spinnerX = new QPiecewiseDoubleSpinBox;
		spinnerX->setMinimumWidth(30);
		spinnerX->setRange(-180.0, 180.0);

		spinnerY = new QPiecewiseDoubleSpinBox;
		spinnerY->setMinimumWidth(30);
		spinnerY->setRange(-180.0, 180.0);

		spinnerZ = new QPiecewiseDoubleSpinBox;
		spinnerZ->setMinimumWidth(30);
		spinnerZ->setRange(-180.0, 180.0);

		layout->addWidget(name, 0, 0);
		layout->addWidget(spinnerX, 0, 2);
		layout->addWidget(spinnerY, 0, 4);
		layout->addWidget(spinnerZ, 0, 6);
		layout->setSpacing(3);

		std::string template_name = field->getTemplateName();

		double rx = 0;
		double ry = 0;
		double rz = 0;
		bool isDouble = false;

		if (template_name == std::string(typeid(Quat1f).name()))
		{
			FVar<Quat1f>* f = TypeInfo::cast<FVar<Quat1f>>(field);
			auto q = f->getValue();
			float yaw, pitch, roll;
			q.toEulerAngle(yaw, pitch, roll);
			rx = glm::degrees(roll);
			ry = glm::degrees(pitch);
			rz = glm::degrees(yaw);
		}
		else if (template_name == std::string(typeid(Quat1d).name()))
		{
			FVar<Quat1d>* f = TypeInfo::cast<FVar<Quat1d>>(field);
			auto q = f->getValue();
			double yaw, pitch, roll;
			q.toEulerAngle(yaw, pitch, roll);
			rx = glm::degrees(roll);
			ry = glm::degrees(pitch);
			rz = glm::degrees(yaw);
			isDouble = true;
		}

		spinnerX->setRealValue(rx);
		spinnerY->setRealValue(ry);
		spinnerZ->setRealValue(rz);

		spinnerX->setDouble(isDouble);
		spinnerY->setDouble(isDouble);
		spinnerZ->setDouble(isDouble);

		QObject::connect(spinnerX, SIGNAL(editingFinishedWithValue(double)), this, SLOT(updateField(double)));
		QObject::connect(spinnerY, SIGNAL(editingFinishedWithValue(double)), this, SLOT(updateField(double)));
		QObject::connect(spinnerZ, SIGNAL(editingFinishedWithValue(double)), this, SLOT(updateField(double)));

		QObject::connect(name, SIGNAL(toggle(bool)), spinnerX, SLOT(toggleDecimals(bool)));
		QObject::connect(name, SIGNAL(toggle(bool)), spinnerY, SLOT(toggleDecimals(bool)));
		QObject::connect(name, SIGNAL(toggle(bool)), spinnerZ, SLOT(toggleDecimals(bool)));

		QObject::connect(this, SIGNAL(fieldChanged()), this, SLOT(updateWidget()));
	}

	QQuatFieldWidget::~QQuatFieldWidget()
	{
	}

	void QQuatFieldWidget::updateField(double)
	{
		if (m_updating) return;

		double rx = spinnerX->getRealValue();
		double ry = spinnerY->getRealValue();
		double rz = spinnerZ->getRealValue();

		double roll = glm::radians(rx);
		double pitch = glm::radians(ry);
		double yaw = glm::radians(rz);

		std::string template_name = field()->getTemplateName();

		if (template_name == std::string(typeid(Quat1f).name()))
		{
			FVar<Quat1f>* f = TypeInfo::cast<FVar<Quat1f>>(field());
			Quat1f q = Quat1f::fromEulerAngles((float)yaw, (float)pitch, (float)roll);
			f->setValue(q);
		}
		else if (template_name == std::string(typeid(Quat1d).name()))
		{
			FVar<Quat1d>* f = TypeInfo::cast<FVar<Quat1d>>(field());
			Quat1d q = Quat1d::fromEulerAngles(yaw, pitch, roll);
			f->setValue(q);
		}

		emit fieldChanged();
	}

	void QQuatFieldWidget::updateWidget()
	{
		std::string template_name = field()->getTemplateName();

		double rx = 0;
		double ry = 0;
		double rz = 0;

		if (template_name == std::string(typeid(Quat1f).name()))
		{
			FVar<Quat1f>* f = TypeInfo::cast<FVar<Quat1f>>(field());
			auto q = f->getValue();
			float yaw, pitch, roll;
			q.toEulerAngle(yaw, pitch, roll);
			rx = glm::degrees(roll);
			ry = glm::degrees(pitch);
			rz = glm::degrees(yaw);
		}
		else if (template_name == std::string(typeid(Quat1d).name()))
		{
			FVar<Quat1d>* f = TypeInfo::cast<FVar<Quat1d>>(field());
			auto q = f->getValue();
			double yaw, pitch, roll;
			q.toEulerAngle(yaw, pitch, roll);
			rx = glm::degrees(roll);
			ry = glm::degrees(pitch);
			rz = glm::degrees(yaw);
		}

		m_updating = true;
		spinnerX->blockSignals(true);
		spinnerY->blockSignals(true);
		spinnerZ->blockSignals(true);

		spinnerX->setRealValue(rx);
		spinnerY->setRealValue(ry);
		spinnerZ->setRealValue(rz);

		spinnerX->blockSignals(false);
		spinnerY->blockSignals(false);
		spinnerZ->blockSignals(false);
		m_updating = false;
	}

	void QQuatFieldWidget::quatValueChange(double)
	{
		double rx = spinnerX->getRealValue();
		double ry = spinnerY->getRealValue();
		double rz = spinnerZ->getRealValue();

		double roll = glm::radians(rx);
		double pitch = glm::radians(ry);
		double yaw = glm::radians(rz);

		Quat1d q = Quat1d::fromEulerAngles(yaw, pitch, roll);
		emit quatChange(q.x, q.y, q.z, q.w);
	}
}
