#include "Module/VirtualModel.h"

namespace dyno
{
    IMPLEMENT_CLASS_1(VirtualModel, TDataType)

    template<typename TDataType>
    VirtualModel<TDataType>::VirtualModel()
    {
    }

    template<typename TDataType>
    VirtualModel<TDataType>::~VirtualModel()
    {
    }

    DEFINE_CLASS(VirtualModel);
    
} // namespace dyno
