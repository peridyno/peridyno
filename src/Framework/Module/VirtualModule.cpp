#include "Module/VirtualModule.h"

namespace dyno
{
    IMPLEMENT_TCLASS(VirtualModule, TDataType)

    template<typename TDataType>
    VirtualModule<TDataType>::VirtualModule()
    {
    }

    template<typename TDataType>
    VirtualModule<TDataType>::~VirtualModule()
    {
    }

    DEFINE_CLASS(VirtualModule);
    
} // namespace dyno
