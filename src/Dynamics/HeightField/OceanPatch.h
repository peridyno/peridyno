#pragma once
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <math_constants.h>
#include "HeightFieldNode.h"

namespace dyno
{

	struct WindParam
	{
		float windSpeed;
		float A;
		float choppiness;
		float global;
	};

	class OceanPatch : public Node
	{
	public:
		OceanPatch(int size, float patchSize, int windType = 8, std::string name = "default");
		OceanPatch(int size, float wind_dir, float windSpeed, float A_p, float max_choppiness, float global);
		~OceanPatch();

		bool initialize() override;

		void animate(float t);

		float getMaxChoppiness();
		float getChoppiness();

		//返回实际覆盖面积，以m为单位
		float getPatchSize() { return m_realPatchSize; }

		//返回网格分辨率
		float getGridSize() { return m_size; }
		float getGlobalShift() { return m_globalShift; }
		float getGridLength() { return m_realPatchSize / m_size; }
		void setChoppiness(float value) { m_choppiness = value; }



		float2* getHeightField() { return m_ht; }
		float4* getDisplacement() { return m_displacement; }
		//GLuint getDisplacementTextureId() { return m_displacement_texture; }
		//GLuint getGradientTextureId() { return m_gradient_texture; }

	public:
		float m_windSpeed = 0;					//风速
		float windDir = CUDART_PI_F / 3.0f;	//风场方向
		int m_windType;			//风力等级，目前设置为0~12
		float m_fft_real_length = 10;
		float m_fft_flow_speed = 1.0f;

		float4* m_displacement = nullptr;		// 位移场
		float4* m_gradient = nullptr;			// gradient field

	private:
		void generateH0(float2* h0);
		float gauss();
		float phillips(float Kx, float Ky, float Vdir, float V, float A, float dir_depend);

		int m_size;

		int m_spectrumW;		//频谱宽度
		int m_spectrumH;		//频谱长度

		float m_choppiness;		//设置浪尖的尖锐性，范围0~1

		std::vector<WindParam> m_params;	//不同风力等级下的FFT变换参数

		const float g = 9.81f;              //重力
		float A = 1e-7f;					//波的缩放系数
		float m_realPatchSize;				//实际覆盖面积，以m为单位
		float dirDepend = 0.07f;			//风长方向相关性

		float m_maxChoppiness;				//设置choppiness上限
		float m_globalShift;				//大尺度偏移幅度

		float2* m_h0;						//初始频谱
		float2* m_ht;						//当前时刻频谱

		float2* m_Dxt;						//x方向偏移
		float2* m_Dzt;						//z方向偏移

		cufftHandle fftPlan;
	};

}
