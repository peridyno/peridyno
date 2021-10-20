#include "ParticleWriterABC.h"

#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcCoreAbstract/Tests/Assert.h>

#include <sstream>
#include <iostream>
#include <fstream>
#include <string>

#include <ImathRandom.h>

namespace dyno
{
	IMPLEMENT_CLASS_1(ParticleWriterABC, TDataType)

	template<typename TDataType>
	ParticleWriterABC<TDataType>::ParticleWriterABC()
	: OutputModule()
	{
	}

	template<typename TDataType>
	ParticleWriterABC<TDataType>::~ParticleWriterABC()
	{
	}

	template<typename TDataType>
	void ParticleWriterABC<TDataType>::flush()
	{
		if (time_idx % 8 != 0)
		{
			time_idx++;
			return;
		}
		time_idx++;

		auto& inPos = this->inPointSet()->getDataPtr()->getPoints();
		auto& inColor = this->inColor()->getData();

		assert(inPos.size() == inColor.size());

		std::stringstream ss; ss << m_output_index;
		std::string filename = this->varOutputPath()->getData() + std::string("fluid_pos_") + ss.str() + std::string(".abc");

		Alembic::AbcGeom::OArchive archive(Alembic::AbcCoreOgawa::WriteArchive(),
			filename);
		Alembic::AbcGeom::OObject topObj(archive, Alembic::AbcGeom::kTop);
		Alembic::AbcGeom::OPoints ptsObj(topObj, "somePoints");

		int total_num = inPos.size();
		CArray<Coord> host_position;
		CArray<Real> host_mapping;
		host_position.resize(total_num);
		host_mapping.resize(total_num);

		host_position.assign(inPos);
		host_mapping.assign(inColor);

		

		std::vector< Alembic::AbcGeom::V3f > positions;
		std::vector< Alembic::AbcGeom::V3f > velocities;
		std::vector< Alembic::Util::uint64_t > ids;
		std::vector< Alembic::Util::float32_t > widths;

		for (int i = 0; i < total_num; i++)
		{
				positions.push_back(Alembic::AbcGeom::V3f(host_position[i][0], host_position[i][1], host_position[i][2]));
				velocities.push_back(Alembic::AbcGeom::V3f(0, 0, 0));
				ids.push_back(i);
				widths.push_back(host_mapping[i]);
			//if(i % 100000 == 0)
			//printf("pressure: %.3lf\n", host_mapping[i]);
		}
		Alembic::AbcGeom::OFloatGeomParam::Sample widthSamp;
		widthSamp.setScope(Alembic::AbcGeom::kVertexScope);
		widthSamp.setVals(Alembic::AbcGeom::FloatArraySample(widths));

		Alembic::AbcGeom::OPointsSchema::Sample psamp(Alembic::AbcGeom::V3fArraySample(positions),
			Alembic::AbcGeom::UInt64ArraySample(ids), Alembic::AbcGeom::V3fArraySample(velocities),
			widthSamp);

		ptsObj.getSchema().set(psamp);

		host_position.clear();
		host_mapping.clear();

		m_output_index++;
	}

	
	DEFINE_CLASS(ParticleWriterABC)
}