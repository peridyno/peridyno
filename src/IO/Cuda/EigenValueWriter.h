#pragma once
#include "Module/OutputModule.h"
#include "Module/TopologyModule.h"
#include "Matrix/Transform3x3.h"

#include <string>

namespace dyno
{

	template <typename TDataType> class TriangleSet;
	template<typename TDataType>
	class EigenValueWriter : public OutputModule
	{
		DECLARE_TCLASS(EigenValueWriter, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;
		typedef typename Transform<Real, 3> Transform;
		EigenValueWriter();
		virtual ~EigenValueWriter();

		void output()override;

	protected:

	public:
		
		DEF_ARRAY_IN(Transform, Transform, DeviceType::GPU, "transform");

	private:
		int mFileIndex = -1;
	};
}
