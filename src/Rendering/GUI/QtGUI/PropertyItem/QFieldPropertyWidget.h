#pragma once
#include "QtGUI/PropertyItem/QFieldWidget.h"

namespace dyno
{
    class QFieldPropertyWidget;
    class FieldSignal: public QObject {
        Q_OBJECT
    public:
        FieldSignal(bool tick, bool update, QObject* parent = nullptr);
    Q_SIGNALS:
        void fieldPromote(FieldSignal*);
    public Q_SLOTS:
        void fieldChanged();

        bool isTick() const;
        bool isUpdate() const;

    private:
        bool mTick;
        bool mUpdate;
    };

	class PERIDYNO_QTGUI_API QFieldPropertyWidget: public QFieldWidget {
		Q_OBJECT
	public:
		QFieldPropertyWidget(FBase* field);
		virtual ~QFieldPropertyWidget();

        QWidget* createFieldWidget(FBase*, bool tick = true, bool update = false);

    public Q_SLOTS:
        void fieldPromote(FieldSignal*);
	};
}
