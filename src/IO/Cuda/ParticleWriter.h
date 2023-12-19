#pragma once
#include "Module/OutputModule.h"
#include "Module/TopologyModule.h"
#include "Topology/PointSet.h"
#include <string>

namespace dyno
{
	template <typename TDataType> class TriangleSet;
	template<typename TDataType>
	class ParticleWriter : public OutputModule
	{
		DECLARE_TCLASS(ParticleWriter, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		DECLARE_ENUM(OpenType,
			ASCII = 0,
			binary = 1);

		ParticleWriter();
		virtual ~ParticleWriter();

		void OutputASCII(std::string filename);
		void OutputBinary(std::string filename);

		void output()override;
	protected:

	public:

		DEF_INSTANCE_IN(PointSet<TDataType>, PointSet, "Input PointSet");
		DEF_ENUM(OpenType, FileType, ASCII, "FileType");

	private:
	};
}
