#ifndef QNODEPROPERTYWIDGET_H
#define QNODEPROPERTYWIDGET_H

#include <QToolBox>
#include <QWidget>
#include <QGroupBox>
#include <QScrollArea>
#include <QGridLayout>
#include <QVBoxLayout>

#include "Nodes/QtBlock.h"

#include <vector>



namespace dyno
{
	class Base;
	class Node;
	class Module;
	class FieldBase;
	class QDoubleSpinner;
	class PVTKOpenGLWidget;

	class QBoolFieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QBoolFieldWidget(FieldBase* field);
		~QBoolFieldWidget() {};

	Q_SIGNALS:
		void fieldChanged();

	public slots:
		void changeValue(int status);

	private:
		FieldBase* m_field = nullptr;
	};

	class QIntegerFieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QIntegerFieldWidget(FieldBase* field);
		~QIntegerFieldWidget() {};

	Q_SIGNALS:
		void fieldChanged();

	public slots:
		void changeValue(int);

	private:
		FieldBase* m_field = nullptr;
	};

	class QRealFieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QRealFieldWidget(FieldBase* field);
		~QRealFieldWidget() {};

	Q_SIGNALS:
		void fieldChanged();

	public slots:
		void changeValue(double);

	private:
		FieldBase* m_field = nullptr;
	};


	class QVector3FieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QVector3FieldWidget(FieldBase* field);
		~QVector3FieldWidget() {};

	Q_SIGNALS:
		void fieldChanged();

	public slots:
		void changeValue(double);

	private:
		FieldBase* m_field = nullptr;

		QDoubleSpinner* spinner1;
		QDoubleSpinner* spinner2;
		QDoubleSpinner* spinner3;
	};

	class PPropertyWidget : public QWidget
	{
		Q_OBJECT
	public:
		explicit PPropertyWidget(QWidget *parent = nullptr);
		~PPropertyWidget();

		virtual QSize sizeHint() const;

//		void clear();

	//signals:
		QWidget* addWidget(QWidget* widget);
		void removeAllWidgets();


	public slots:
		void showProperty(Module* module);
		void showProperty(Node* node);

		void showBlockProperty(QtNodes::QtBlock& block);

		void updateDisplay();

	private:
		void updateContext(Base* base);

		void addScalarFieldWidget(FieldBase* field);
		void addArrayFieldWidget(FieldBase* field);

		QVBoxLayout* m_main_layout;
		QScrollArea* m_scroll_area;
		QWidget * m_scroll_widget;
		QGridLayout* m_scroll_layout;

		std::vector<QWidget*> m_widgets;
	};

}

#endif // QNODEPROPERTYWIDGET_H
