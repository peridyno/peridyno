#include "CharacterController.h"

namespace dyno
{
	IMPLEMENT_TCLASS(CharacterController, TDataType);

	template<typename TDataType>
	CharacterController<TDataType>::CharacterController()
		: KeyboardInputModule()
	{
		this->varCacheEvent()->setValue(false);
	}

	template<typename TDataType>
	void CharacterController<TDataType>::onEvent(PKeyboardEvent event)
	{

		
		Vec3f dir = Vec3f(0);
	

		switch (event.key)
		{
		case PKeyboardType::PKEY_W:
			dir.x = -1;

			break;

		case PKeyboardType::PKEY_S:
			dir.x = 1;
			break;

		case PKeyboardType::PKEY_A:
			dir.z = 1;
			break;

		case PKeyboardType::PKEY_D:
			dir.z = -1;
			break;

		}
		
		mDispatcher.callDispatcher("setAxisValue",dir.normalize());


	


	}

	DEFINE_CLASS(CharacterController);
}


