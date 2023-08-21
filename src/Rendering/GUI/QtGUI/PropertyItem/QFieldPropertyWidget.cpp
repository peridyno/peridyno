#include "QtGUI/PropertyItem/QFieldPropertyWidget.h"
#include "QtGUI/PPropertyWidget.h"
#include "Field.h"

using namespace dyno;

QFieldPropertyWidget::QFieldPropertyWidget(dyno::FBase* field): dyno::QFieldWidget(field) {
}
QFieldPropertyWidget::~QFieldPropertyWidget() {}

QWidget* QFieldPropertyWidget::createFieldWidget(dyno::FBase* f, bool tick, bool update) {
    auto sig = new FieldSignal(tick, update, this);
    auto fw = dyno::PPropertyWidget::createFieldWidget(f);

    connect(fw, SIGNAL(fieldChanged()), sig, SLOT(fieldChanged()));
    connect(sig, &FieldSignal::fieldPromote, this, &QFieldPropertyWidget::fieldPromote);
    return fw;
}

void QFieldPropertyWidget::fieldPromote(FieldSignal* s) {
    if(s->isTick())
        field()->tick();
    if(s->isUpdate())
        field()->update();
}

FieldSignal::FieldSignal(bool tick, bool update, QObject* parent):QObject(parent),mTick(tick),mUpdate(update) {}

bool FieldSignal::isTick() const {
    return mTick;
}

bool FieldSignal::isUpdate() const {
    return mUpdate;
}

void FieldSignal::fieldChanged() {
    emit fieldPromote(this);
}
