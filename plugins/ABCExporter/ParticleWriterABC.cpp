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
	IMPLEMENT_TCLASS(ParticleWriterABC, TDataType)

	template<typename TDataType>
	ParticleWriterABC<TDataType>::ParticleWriterABC()
	: OutputModule()
	{
		this->inPointSet()->tagOptional(true);
		this->inColor()->tagOptional(true);
		this->inPosition()->tagOptional(true);

	}

	template<typename TDataType>
	ParticleWriterABC<TDataType>::~ParticleWriterABC()
	{
	}

	template<typename TDataType>
	void ParticleWriterABC<TDataType>::output()
	{
		if (time_idx % this->varInterval()->getValue() != 0)
		{
			time_idx++;
			return;
		}
		time_idx++;

		std::cout << "............Writer .abc file............" << std::endl;
		DArray<Vec3f>* inPos = nullptr;
		if (!this->inPointSet()->isEmpty())
			inPos = &this->inPointSet()->constDataPtr()->getPoints();
		else if (!this->inPosition()->isEmpty())
			inPos = &this->inPosition()->getData();
		else 
		{
			std::cout << "Error : No inPointSet or inPosition input!!\n";
			return;
		}

		DArray<Real>* inColor = nullptr;
		if (!this->inColor()->isEmpty())
			inColor = &this->inColor()->getData();

		if (inColor != nullptr)
			if (inColor->size() != inColor->size()) 
			{
				std::cout << "Error inColor Input!!\n";
				return;
			}


		std::stringstream ss; ss << m_output_index;
		std::string filename = this->varOutputPath()->getValue().string() + std::string("fluid_pos_") + ss.str() + std::string(".abc");

		Alembic::AbcGeom::OArchive archive(Alembic::AbcCoreOgawa::WriteArchive(),
			filename);
		Alembic::AbcGeom::OObject topObj(archive, Alembic::AbcGeom::kTop);
		Alembic::AbcGeom::OPoints ptsObj(topObj, "somePoints");

		//CPU Data
		int total_num = inPos->size();
		CArray<Coord> host_position;
		host_position.resize(total_num);
		host_position.assign(*inPos);

		CArray<Real> host_mapping;
		if (inColor != nullptr) 
		{
			host_mapping.resize(total_num);
			host_mapping.assign(*inColor);
		}

		//Alembic Data
		std::vector< Alembic::AbcGeom::V3f > positions;
		std::vector< Alembic::AbcGeom::V3f > velocities;
		std::vector< Alembic::Util::uint64_t > ids;
		std::vector< Alembic::Util::float32_t > widths;

		for (int i = 0; i < total_num; i++)
		{
				positions.push_back(Alembic::AbcGeom::V3f(host_position[i][0], host_position[i][1], host_position[i][2]));
				velocities.push_back(Alembic::AbcGeom::V3f(0, 0, 0));
				ids.push_back(i);
				if (inColor != nullptr)
					widths.push_back(host_mapping[i]);
				else
					widths.push_back(Alembic::Util::float32_t(0));
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