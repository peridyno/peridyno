#pragma once
#include "Node.h"

#include "FilePath.h"
#include "Node/ParametricModel.h"

namespace dyno
{
	/*!
	*	\class	Geometry Loader
	*	\brief	Base class for other geometry loaders
	*/
	template<typename TDataType>
	class GeometryLoader : public ParametricModel<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		GeometryLoader();

	public:
		DEF_VAR(FilePath, FileName, "", "");
	};
}