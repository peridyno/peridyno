#pragma once
#include "Framework/ModuleIO.h"
#include "Framework/ModuleTopology.h"

#include <string>

namespace dyno
{

	template <typename TDataType> class TriangleSet;
	template<typename TDataType>
	class ParticleWriter : public IOModule
	{
		DECLARE_CLASS_1(ParticleWriter, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;
		ParticleWriter();
		virtual ~ParticleWriter();

		void setNamePrefix(std::string prefix);
		void setOutputPath(std::string path);

	protected:
		void updateImpl() override;

	public:
		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "");

	private:
		int mFileIndex = 0;

		std::string mOutpuPath;
		std::string mOutputPrefix;
	};
}
