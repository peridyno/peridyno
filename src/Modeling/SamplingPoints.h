
#pragma once
#include "Node.h"

#include "Topology/Primitive3D.h"
#include "Topology/PointSet.h"

#include "GLPointVisualModule.h"

namespace dyno
{
	template<typename TDataType>
	class SamplingPoints : public Node
	{
		DECLARE_TCLASS(SamplingPoints, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SamplingPoints();

		void disableRender();

		int pointSize();

	public:

		DEF_INSTANCE_STATE(PointSet<TDataType>, PointSet, "");

	private:
		std::shared_ptr <GLPointVisualModule> glModule;
	};

	IMPLEMENT_TCLASS(SamplingPoints, TDataType);
}