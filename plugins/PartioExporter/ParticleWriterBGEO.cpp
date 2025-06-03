#include "ParticleWriterBGEO.h"

#include "Partio.h"

#include <string>


namespace dyno
{
	IMPLEMENT_TCLASS(ParticleWriterBGEO, TDataType)

	template<typename TDataType>
	ParticleWriterBGEO<TDataType>::ParticleWriterBGEO()
	: OutputModule()
	{
		this->inAttribute0()->tagOptional(true);
		this->inAttribute1()->tagOptional(true);
		this->inAttribute2()->tagOptional(true);
		this->inAttribute3()->tagOptional(true);
		this->inAttribute4()->tagOptional(true);
	}

	template<typename TDataType>
	ParticleWriterBGEO<TDataType>::~ParticleWriterBGEO()
	{
		m_positions.clear();
		for (int i = 0; i < 5; ++i)
			m_attributes[i].clear();
	}

	template<typename TDataType>
	void ParticleWriterBGEO<TDataType>::output()
	{
		const int MaxAttrNum = 5;

		std::string atterNames[MaxAttrNum] = {this->varAttributeName0()->getValue(), 
												this->varAttributeName1()->getValue(),
												this->varAttributeName2()->getValue(),
												this->varAttributeName3()->getValue(),
												this->varAttributeName4()->getValue()};	
		
		std::shared_ptr<DArray<Real>> attersPtr[MaxAttrNum] = {nullptr, nullptr, nullptr, nullptr, nullptr};
		if (!this->inAttribute0()->isEmpty()) attersPtr[0] = this->inAttribute0()->getDataPtr();
		if (!this->inAttribute1()->isEmpty()) attersPtr[1] = this->inAttribute1()->getDataPtr();
		if (!this->inAttribute2()->isEmpty()) attersPtr[2] = this->inAttribute2()->getDataPtr();
		if (!this->inAttribute3()->isEmpty()) attersPtr[3] = this->inAttribute3()->getDataPtr();
		if (!this->inAttribute4()->isEmpty()) attersPtr[4] = this->inAttribute4()->getDataPtr();

		std::string outputFileName = this->constructFileName() + ".bgeo";
													
		printf(">>>>>>>>>>>>>>>>>> Opening output bgeo file %s <<<<<<<<<<<<<<<<<<\n", outputFileName.c_str());

		// Copy Data from GPU
		int AttNum = this->varAttributeNum()->getValue();
		m_positions.assign(this->inPosition()->getData());
		int N = m_positions.size();
		for (int i = 0; i < AttNum; ++i)
			m_attributes[i].assign(*attersPtr[i]);
		
		// Init partio
		Partio::ParticlesDataMutable* parts = Partio::create();
		Partio::ParticleAttribute posH;
		Partio::ParticleAttribute attrH[MaxAttrNum];

		posH = parts->addAttribute("position", Partio::VECTOR, 3);
		for (int i = 0; i < AttNum; ++i)
			attrH[i]= parts->addAttribute(atterNames[i].c_str(), Partio::VECTOR, 1);
		
		// write to partio structure
		for (int k = 0; k < N; k++) {
			int idx = parts->addParticle();
			float* posP = parts->dataWrite<float>(posH, idx);
			float* attrP[MaxAttrNum];
			for (int i = 0; i < AttNum; i++)
				attrP[i] = parts->dataWrite<float>(attrH[i], idx);
			
			// write pos
			for (int d = 0; d < 3; ++d)
				posP[d] = 0;
			for (int d = 0; d < 3; ++d)
				posP[d] = (float)m_positions[k][d];

			// write attributes
			for (int i = 0; i < AttNum; i++)
			{
				attrP[i][0] = 0;
				attrP[i][0] = (float)m_attributes[i][k];
			}
		}

		Partio::write(outputFileName.c_str(), *parts);
		parts->release();

		printf(">>>>>>>>>>>>>>>>>> Ouput Attribute { position ");
		for (int i = 0; i < AttNum; ++i)
			printf("%s ", atterNames[i].c_str());
		printf("} <<<<<<<<<<<<<<<<<<\n");
		printf(">>>>>>>>>>>>>>>>>> Finish output %d particles <<<<<<<<<<<<<<<<<<\n", N);

	}

	
	DEFINE_CLASS(ParticleWriterBGEO)
}