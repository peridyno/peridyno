#include "QArrayVar.h"

void dyno::ArrayWidgetBase::connectSignal(QFieldWidget* field, int id)
{
    QObject::connect(field, QOverload<>::of(&QFieldWidget::fieldChanged), [this, id]()
        {
            OnElementUpdate(id);
        }
    );   
}
