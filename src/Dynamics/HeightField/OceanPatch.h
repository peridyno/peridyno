#pragma once
#include "Node.h"

#include <cufft.h>
#include <math_constants.h>

namespace dyno {

    struct WindParam
    {
        float windSpeed;
        float A;
        float choppiness;
        float global;
    };

    template<typename TDataType>
    class OceanPatch : public Node
    {
        DECLARE_TCLASS(OceanPatch, TDataType)
    public:
        typedef typename dyno::Vector<float, 2> Coord;

        OceanPatch(int size, float patchSize, int windType = 1, std::string name = "OceanPatch");
        OceanPatch(int size, float wind_dir, float windSpeed, float A_p, float max_choppiness, float global);
        OceanPatch(std::string name = "OceanPatch");
        ~OceanPatch();

        void animate(float t);

        float getMaxChoppiness();
        float getChoppiness();

        void resetWindType();

        //����ʵ�ʸ����������mΪ��λ
        float getPatchSize()
        {
            return m_realPatchSize;
        }

        //��������ֱ���
        float getGridSize()
        {
            return mResolution;
        }
        float getGlobalShift()
        {
            return m_globalShift;
        }
        float getGridLength()
        {
            return m_realPatchSize / mResolution;
        }
        void setChoppiness(float value)
        {
            mChoppiness = value;
        }

	    DArray2D<Coord> getHeightField()
        {
            return m_ht;
        }
        DArray2D <Vec4f> getDisplacement()
        {
            return m_displacement;
        }

        DEF_INSTANCE_STATE(TopologyModule, Topology, "Topology");

    public:
        float m_windSpeed = 0;                   //����
        float windDir     = CUDART_PI_F / 3.0f;  //�糡����
        int   m_windType;                        //�����ȼ���Ŀǰ����Ϊ0~12
        float m_fft_real_length = 10;
        float m_fft_flow_speed  = 1.0f;

        DArray2D<Vec4f> m_displacement;  // λ�Ƴ�
        DArray2D<Vec4f> m_gradient;      // gradient field

    protected:
	    void resetStates() override;
	    void updateStates() override;
	    void updateTopology() override;
        
    private:
        void  generateH0(DArray2D<Coord> h0);
        float gauss();
        float phillips(float Kx, float Ky, float Vdir, float V, float A, float dir_depend);

        int mResolution;

        int mSpectrumWidth;  //Ƶ�׿���
        int mSpectrumHeight;  //Ƶ�׳���

        float mChoppiness;  //�����˼�ļ����ԣ���Χ0~1

        std::vector<WindParam> m_params;  //��ͬ�����ȼ��µ�FFT�任����

        const float g = 9.81f;          //����
        float       A = 1e-7f;          //��������ϵ��
        float       m_realPatchSize;    //ʵ�ʸ����������mΪ��λ
        float       dirDepend = 0.07f;  //�糤���������

        float m_maxChoppiness;  //����choppiness����
        float m_globalShift;    //��߶�ƫ�Ʒ���

        DArray2D<Coord> m_h0;  //��ʼƵ��
        DArray2D<Coord> m_ht;  //��ǰʱ��Ƶ��

        DArray2D<Coord> m_Dxt;  //x����ƫ��
        DArray2D<Coord> m_Dzt;  //z����ƫ��

        cufftHandle fftPlan;


        DEF_VAR(int, my_windTypes, 4, "m_windTypesWinds");
    };
    IMPLEMENT_TCLASS(OceanPatch, TDataType)
} 
