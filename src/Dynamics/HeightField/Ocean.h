#pragma once
#include "OceanPatch.h"
#include "CapillaryWave.h"
namespace dyno
{
	template<typename TDataType>
	class Ocean : public Node
	{
		DECLARE_TCLASS(Ocean, TDataType)
	public:
		Ocean();
		~Ocean();

		void animate(float dt);

		float2 getWindDirection() { 
			auto m_patch = this->getOceanPatch();
			return make_float2(cosf(m_patch->windDir), sinf(m_patch->windDir)); 
		}
		
		float getFftRealSize() { return m_fft_real_size; }
		int getFftResolution() { return m_fft_size; }
		float getChoppiness() { return m_choppiness; }

		void setChoppiness(float chopiness) {
			m_choppiness = chopiness;
			this->getOceanPatch()->setChoppiness(m_choppiness);
		}
		//OceanPatch* getOceanPatch() { return m_patch; }
		int getGridSize() { return m_fft_size; }

		//返回每个块实际覆盖的距离
		float getPatchLength();
		float getGridLength();

		DEF_NODE_PORT(OceanPatch<TDataType>, OceanPatch, "Ocean Patch");
		DEF_NODE_PORTS(CapillaryWave, CapillaryWave<TDataType>, "Capillary Wave");

	protected:
		void resetStates() override;
		void updateStates() override;

	public:
		//挤压波形形成尖浪
		float m_choppiness = 1.0f;
		
		//初始风级
		//int m_windType = 8;

		//fft纹理分辨率
		int m_fft_size = 512;

		//fft纹理贴图的物理长度 单位米
		float m_fft_real_size = 2.0f;

		//高度场贴图的物理长度 单位米
		float m_heightfield_real_size = 5000;

		int m_oceanWidth;
		int m_oceanHeight;

		float m_eclipsedTime;

		float m_patchSize = 512;
		float m_virtualGridSize;
		float m_realGridSize;

		int Nx = 1;
		int Ny = 1;


		
	};
	IMPLEMENT_TCLASS(Ocean, TDataType)
}