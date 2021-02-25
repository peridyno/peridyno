/**
 * @file HyperelasticityModule.h
 * @author Xiaowei He (xiaowei@iscas.ac.cn)
 * @brief This is an implementation of hyperelasticity based on a set of basis functions.
 * 		  For more details, please refer to [Xu et al. 2018] "Reformulating Hyperelastic Materials with Peridynamic Modeling"
 * @version 0.1
 * @date 2019-06-18
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#pragma once
#include "ElasticityModule.h"
#include "Attribute.h"

namespace dyno {

//#define STVK_MODEL
#define LINEAR_MODEL
//#define NEOHOOKEAN_MODEL
//#define XU_MODEL
	/**
	 * @brief Basis Function
	 * 
	 * @tparam T Real value
	 * @tparam n The degree of the basis
	 */

	/***********************************************************************************************
	template<typename T, int n>
	class Basis
	{
	public:
		static DYN_FUNC T A(T s) {
			return ((pow(s, n + 1) - 1) / (n + 1) + (pow(s, 1 - n) - 1) / (n - 1)) / n;
		}
		
		static DYN_FUNC T B(T s) {
			return 2 * ((pow(s, n + 1) - 1) / (n + 1) - s + 1) / n;
		}

		static DYN_FUNC T dA(T s) {
			Real sn = pow(s, n);
			return (sn - 1 / sn) / 2;
		}

		static DYN_FUNC T dB(T s) {
			return 2 * (pow(s, n) - 1);
		}
	};

	template<typename T>
	class Basis<T, 1>
	{
	public:
		static DYN_FUNC T A(T s) {
			return (s * s - 1) / 2 - log(s);
		}

		static	DYN_FUNC T B(T s) {
			return s * s - s;
		}

		static DYN_FUNC T dA(T s) {
			return s - 1 / s;
		}

		static DYN_FUNC T dB(T s) {
			return 2 * (s - 1);
		}
	};
	**************************************************************************/

	enum EnergyType
	{
		StVK,
		NeoHooekean,
		Polynomial,
		Xuetal
	};

	template<typename Real, typename Matrix>
	class HyperelasticityModel
	{
	public:
		DYN_FUNC HyperelasticityModel() {};

		DYN_FUNC virtual Real getEnergy(Real lambda1, Real lambda2, Real lambda3) = 0;
		DYN_FUNC virtual Matrix getStressTensorPositive(Real lambda1, Real lambda2, Real lambda3) = 0;
		DYN_FUNC virtual Matrix getStressTensorNegative(Real lambda1, Real lambda2, Real lambda3) = 0;

		Real density;
	};

	template<typename Real, typename Matrix>
	class LinearModel : public HyperelasticityModel<Real, Matrix>
	{
	public:
		DYN_FUNC LinearModel() : HyperelasticityModel<Real, Matrix>()
		{
			density = Real(1000);
			s1 = Real(48000);
			s0 = Real(12000);
		}

		DYN_FUNC virtual Real getEnergy(Real lambda1, Real lambda2, Real lambda3) override
		{
			return s0*(lambda1 - 1)*(lambda1 - 1) + s0 * (lambda2 - 1)*(lambda2 - 1) + s0 * (lambda3 - 1)*(lambda3 - 1);
		}

		DYN_FUNC virtual Matrix getStressTensorPositive(Real lambda1, Real lambda2, Real lambda3) override
		{
			return s0 * Matrix::identityMatrix();
		}

		DYN_FUNC virtual Matrix getStressTensorNegative(Real lambda1, Real lambda2, Real lambda3) override
		{
			return s0 * Matrix::identityMatrix();
		}

		Real s0;
		Real s1;
	};


	template<typename Real, typename Matrix>
	class StVKModel : public HyperelasticityModel<Real, Matrix>
	{
	public:
		DYN_FUNC StVKModel() : HyperelasticityModel<Real, Matrix>()
		{
			density = Real(1000);
			s1 = Real(48000);
			s0 = Real(12000);
		}

		DYN_FUNC virtual Real getEnergy(Real lambda1, Real lambda2, Real lambda3) override
		{
			Real I = lambda1*lambda1 + lambda2*lambda2 + lambda3*lambda3;
			Real sq1 = lambda1*lambda1;
			Real sq2 = lambda2*lambda2;
			Real sq3 = lambda3*lambda3;
			Real II = sq1*sq1 + sq2*sq2 + sq3*sq3;
			return 0.5*s0*(I - 3)*(I - 3) + 0.25*s1*(II - 2 * I + 3);
		}

		DYN_FUNC virtual Matrix getStressTensorPositive(Real lambda1, Real lambda2, Real lambda3) override
		{
			Real I = lambda1*lambda1 + lambda2*lambda2 + lambda3*lambda3;

			Real D1 = 2 * s0*I + s1*lambda1*lambda1;
			Real D2 = 2 * s0*I + s1*lambda2*lambda2;
			Real D3 = 2 * s0*I + s1*lambda3*lambda3;

			Matrix D;
			D(0, 0) = D1;
			D(1, 1) = D2;
			D(2, 2) = D3;
			return D;
		}

		DYN_FUNC virtual Matrix getStressTensorNegative(Real lambda1, Real lambda2, Real lambda3) override
		{
			Matrix D = (6 * s0 + s1)*Matrix::identityMatrix();
			D(0, 0) *= lambda1;
			D(1, 1) *= lambda2;
			D(2, 2) *= lambda3;
			return D;
		}

		Real s0;
		Real s1;
	};

	template<typename Real, typename Matrix>
	class NeoHookeanModel : public HyperelasticityModel<Real, Matrix>
	{
	public:
		DYN_FUNC NeoHookeanModel() : HyperelasticityModel<Real, Matrix>()
		{
			density = Real(1000);
			s1 = Real(48000);
			s0 = Real(12000);
		}

		DYN_FUNC virtual Real getEnergy(Real lambda1, Real lambda2, Real lambda3) override
		{
			Real I = lambda1*lambda1+lambda2*lambda2+lambda3*lambda3;
			Real sqrtIII = lambda1*lambda2*lambda3;
			return s0*(I-3-2*log(sqrtIII))+s1*(sqrtIII-1)*(sqrtIII - 1);
		}

		DYN_FUNC virtual Matrix getStressTensorPositive(Real lambda1, Real lambda2, Real lambda3) override
		{
			Real sq1 = lambda1*lambda1;
			Real sq2 = lambda2*lambda2;
			Real sq3 = lambda3*lambda3;

			Real D1 = 2 * s0 + 2 * s1*sq2*sq3;
			Real D2 = 2 * s0 + 2 * s1*sq3*sq1;
			Real D3 = 2 * s0 + 2 * s1*sq1*sq2;

			Matrix D;
			D(0, 0) = D1;
			D(1, 1) = D2;
			D(2, 2) = D3;
			return D;
		}

		DYN_FUNC virtual Matrix getStressTensorNegative(Real lambda1, Real lambda2, Real lambda3) override
		{
			Real sqrtIII = 1/(lambda1*lambda2*lambda3);
			Real constIII = s0*sqrtIII*sqrtIII + s1*sqrtIII;

			Real sq1 = lambda1*lambda1;
			Real sq2 = lambda2*lambda2;
			Real sq3 = lambda3*lambda3;

			Matrix D;
			D(0, 0) = 2*sq2*sq3*constIII*lambda1;
			D(1, 1) = 2*sq3*sq1*constIII*lambda2;
			D(2, 2) = 2*sq1*sq2*constIII*lambda3;

			return D;
		}

		Real s0;
		Real s1;
	};

	template<typename Real, typename Matrix, int n>
	class PolynomialModel : public HyperelasticityModel<Real, Matrix>
	{
	public:
		DYN_FUNC PolynomialModel() : HyperelasticityModel<Real, Matrix>()
		{
			density = Real(1000);
			
			for (int i = 0; i <= n; i++)
			{
				for (int j = 0; j <= n; j++)
				{
					C[i][j] = Real(12000);
				}
			}
		}

		DYN_FUNC virtual Real getEnergy(Real lambda1, Real lambda2, Real lambda3) override
		{
			Real sq1 = lambda1*lambda1;
			Real sq2 = lambda2*lambda2;
			Real sq3 = lambda3*lambda3;
			Real I1 = sq1 + sq2 + sq3;
			Real I2 = sq1*sq2 + sq2*sq3 + sq3*sq1;

			Real totalE = Real(0);
			for (int i = 0; i <= n; i++)
			{
				for (int j = 0; j <= n; j++)
				{
					totalE += C[i][j] * pow(I1 - 3, Real(i))*pow(I2 - 3, Real(j));
				}
			}

			return totalE;
		}

		DYN_FUNC virtual Matrix getStressTensorPositive(Real lambda1, Real lambda2, Real lambda3) override
		{
			Real sq1 = lambda1*lambda1;
			Real sq2 = lambda2*lambda2;
			Real sq3 = lambda3*lambda3;
			Real I1 = sq1 + sq2 + sq3;
			Real I2 = sq1*sq2 + sq2*sq3 + sq3*sq1;

			Real D1 = Real(0);
			Real D2 = Real(0);
			Real D3 = Real(0);
			for (int i = 0; i <= n; i++)
			{
				for (int j = 0; j <= n; j++)
				{
					int i_minus_one = i - 1 < 0 ? 0 : i - 1;
					int j_minus_one = j - 1 < 0 ? 0 : j - 1;

					int i_minus_one_even = 2 * floor(i_minus_one / 2);
					int j_minus_one_even = 2 * floor(j_minus_one / 2);
					int i_even = 2 * floor(i / 2);
					int j_even = 2 * floor(j / 2);

					Real C1_positive;
					if (i_minus_one_even == i_minus_one && j_even == j)
					{
						C1_positive = Real(1);
					}
					else if (i_minus_one_even != i_minus_one && j_even == j)
					{
						C1_positive = I1;
					}
					else if (i_minus_one_even == i_minus_one && j_even != j)
					{
						C1_positive = I2;
					}
					else
					{
						C1_positive = I1*I2 + 9;
					}

					Real C2_positive;
					if (i_even == i && j_minus_one_even == j_minus_one)
					{
						C2_positive = Real(1);
					}
					else if (i_even != i && j_minus_one_even == j_minus_one)
					{
						C2_positive = I1;
					}
					else if (i_even == i && j_minus_one_even != j_minus_one)
					{
						C2_positive = I2;
					}
					else
					{
						C2_positive = I1*I2 + 9;
					}
					
					int exp_i = i < 0 ? 1 : i;
					int exp_j = j < 0 ? 1 : j;
					D1 += 2 * exp_i * C[i][j] * pow(I1 - 3, Real(i_minus_one_even))*pow(I2 - 3, Real(j_even)) * C1_positive + 4 * exp_j * C[i][j] * pow(I1 - 3, Real(i_even))*pow(I2 - 3, Real(j_minus_one_even))*C2_positive*sq1;
					D2 += 2 * exp_i * C[i][j] * pow(I1 - 3, Real(i_minus_one_even))*pow(I2 - 3, Real(j_even)) * C1_positive + 4 * exp_j * C[i][j] * pow(I1 - 3, Real(i_even))*pow(I2 - 3, Real(j_minus_one_even))*C2_positive*sq2;
					D3 += 2 * exp_i * C[i][j] * pow(I1 - 3, Real(i_minus_one_even))*pow(I2 - 3, Real(j_even)) * C1_positive + 4 * exp_j * C[i][j] * pow(I1 - 3, Real(i_even))*pow(I2 - 3, Real(j_minus_one_even))*C2_positive*sq3;

				}
			}

			Matrix D;
			D(0, 0) = D1*lambda1;
			D(1, 1) = D2*lambda2;
			D(2, 2) = D3*lambda3;

			return D;
		}

		DYN_FUNC virtual Matrix getStressTensorNegative(Real lambda1, Real lambda2, Real lambda3) override
		{
			Real sq1 = lambda1*lambda1;
			Real sq2 = lambda2*lambda2;
			Real sq3 = lambda3*lambda3;
			Real I1 = sq1 + sq2 + sq3;
			Real I2 = sq1*sq2 + sq2*sq3 + sq3*sq1;

			Real D1 = Real(0);
			Real D2 = Real(0);
			Real D3 = Real(0);
			for (int i = 0; i <= n; i++)
			{
				for (int j = 0; j <= n; j++)
				{
					int i_minus_one = i - 1 < 0 ? 0 : i - 1;
					int j_minus_one = j - 1 < 0 ? 0 : j - 1;

					int i_minus_one_even = 2 * floor(i_minus_one / 2);
					int j_minus_one_even = 2 * floor(j_minus_one / 2);
					int i_even = 2 * floor(i / 2);
					int j_even = 2 * floor(j / 2);

					Real C1_positive;
					if (i_minus_one_even == i_minus_one && j_even == j)
					{
						C1_positive = 0;
					}
					else if (i_minus_one_even != i_minus_one && j_even == j)
					{
						C1_positive = 3;
					}
					else if (i_minus_one_even == i_minus_one && j_even != j)
					{
						C1_positive = 3;
					}
					else
					{
						C1_positive = 3 * I1 + 3 * I2;
					}

					Real C2_positive;
					if (i_even == i && j_minus_one_even == j_minus_one)
					{
						C2_positive = 0;
					}
					else if (i_even != i && j_minus_one_even == j_minus_one)
					{
						C2_positive = 3;
					}
					else if (i_even == i && j_minus_one_even != j_minus_one)
					{
						C2_positive = 3;
					}
					else
					{
						C2_positive = 3 * I1 + 3 * I2;
					}

					int exp_i = i < 0 ? 1 : i;
					int exp_j = j < 0 ? 1 : j;
					D1 += 2 * exp_i * C[i][j] * pow(I1 - 3, Real(i_minus_one_even))*pow(I2 - 3, Real(j_even)) * C1_positive + 4 * exp_j * C[i][j] * pow(I1 - 3, Real(i_even))*pow(I2 - 3, Real(j_minus_one_even))*C2_positive*sq1;
					D2 += 2 * exp_i * C[i][j] * pow(I1 - 3, Real(i_minus_one_even))*pow(I2 - 3, Real(j_even)) * C1_positive + 4 * exp_j * C[i][j] * pow(I1 - 3, Real(i_even))*pow(I2 - 3, Real(j_minus_one_even))*C2_positive*sq2;
					D3 += 2 * exp_i * C[i][j] * pow(I1 - 3, Real(i_minus_one_even))*pow(I2 - 3, Real(j_even)) * C1_positive + 4 * exp_j * C[i][j] * pow(I1 - 3, Real(i_even))*pow(I2 - 3, Real(j_minus_one_even))*C2_positive*sq3;

				}
			}

			Matrix D;
			D(0, 0) = D1;
			D(1, 1) = D2;
			D(2, 2) = D3;

			return D;
		}

		Real C[n+1][n+1];
	};


	template<typename Real, typename Matrix>
	class XuModel : public HyperelasticityModel<Real, Matrix>
	{
	public:
		DYN_FUNC XuModel() : HyperelasticityModel<Real, Matrix>()
		{
			density = Real(1000);
			s0 = Real(48000);
		}

		DYN_FUNC virtual Real getEnergy(Real lambda1, Real lambda2, Real lambda3) override
		{
			Real sq1 = lambda1*lambda1;
			Real sq2 = lambda2*lambda2;
			Real sq3 = lambda3*lambda3;
			Real E1 = ((sq1*sq1 - 1) / 4 + (1 / sq1 - 1) / 2) / 3;
			Real E2 = ((sq2*sq2 - 1) / 4 + (1 / sq2 - 1) / 2) / 3;
			Real E3 = ((sq3*sq3 - 1) / 4 + (1 / sq3 - 1) / 2) / 3;

			return s0*(E1 + E2 + E3);
		}

		DYN_FUNC virtual Matrix getStressTensorPositive(Real lambda1, Real lambda2, Real lambda3) override
		{
			Matrix D;
			D(0, 0) = s0*(lambda1*lambda1) / 3;
			D(1, 1) = s0*(lambda2*lambda2) / 3;
			D(2, 2) = s0*(lambda3*lambda3) / 3;
			return D;
		}

		DYN_FUNC virtual Matrix getStressTensorNegative(Real lambda1, Real lambda2, Real lambda3) override
		{
			Matrix D;
			D(0, 0) = lambda1 * s0 / (lambda1*lambda1*lambda1*lambda1) / 3;
			D(1, 1) = lambda2 * s0 / (lambda2*lambda2*lambda2*lambda2) / 3;
			D(2, 2) = lambda3 * s0 / (lambda3*lambda3*lambda3*lambda3) / 3;
			return D;
		}

		Real s0;
	};

	template<typename TDataType>
	class HyperelasticityModule_test : public ElasticityModule<TDataType>
	{
		DECLARE_CLASS_1(ElasticityModule, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef TPair<TDataType> NPair;

		HyperelasticityModule_test();
		~HyperelasticityModule_test() override {};

		/**
		 * @brief Set the energy function
		 * 
		 */
		void setEnergyFunction(EnergyType type) { m_energyType = type; }

		void solveElasticity() override;

		void solveElasticityImplicit();

		void solveElasticityGradientDescent();

		void setAlphaStepCompute(bool value) { this->isAlphaCompute = value; }
		void setChebyshevAcceleration(bool value) { this->isChebyshevAcce = value; }

		Real x_border = 0.5;
		Real max_theta = glm::pi<Real>();
		Real theta = 0.0;
		Real angular_velocity = 0.03;
		Coord rotation_center_point;

		bool release_adjust_points_reachTargetPlace = true;

	public:
		DEF_EMPTY_IN_ARRAY(Attribute, Attribute, DeviceType::GPU, "");

		DEF_EMPTY_IN_ARRAY(Rotation, Matrix, DeviceType::GPU, "");

		DEF_EMPTY_IN_ARRAY(Volume, Real, DeviceType::GPU, "");

	protected:
		bool initializeImpl() override;

		void begin() override;

		//void initializeVolume();

		//void previous_enforceElasticity();

	private:
		void getEnergy(Real& totalEnergy, DeviceArray<Coord>& position);

		EnergyType m_energyType;

		DeviceArray<Real> m_fraction;

		DeviceArray<Real> m_energy;
		DeviceArray<Real> m_alpha;
		DeviceArray<Coord> m_gradient;

		DeviceArray<Coord> m_eigenValues;

		DeviceArray<Matrix> m_F;
		DeviceArray<Matrix> m_invF;
		DeviceArray<bool> m_validOfK;
		DeviceArray<Matrix> m_invK;
		DeviceArray<Matrix> m_matU;
		DeviceArray<Matrix> m_matV;

		//DeviceArray<Coord> y_pre;
		DeviceArray<Coord> y_current;
		DeviceArray<Coord> y_next;
		DeviceArray<Coord> y_residual;
		DeviceArray<Coord> y_gradC;

		DeviceArray<Coord> m_source;
		DeviceArray<Matrix> m_A;

		Reduction<Real>* m_reduce;
		Arithmetic<Real>* m_alg;

		DeviceArray<bool> m_bFixed;
		DeviceArray<int> m_points_move_type;
		DeviceArray<Coord> m_fixedPos;

		bool isAlphaCompute = true;		// compute alpha step length or not
		bool isChebyshevAcce = false;	// chebyshev accelerate or not

		DEF_VAR(isConvergeComputeField, bool, false, "calculate convergency every time step or not")
		DEF_VAR(convergencyEpsilonField, Real, 0.05, "standard epsilon to ensure iteration converge")

		std::map<int, int> iteration_curve;
		int iteration_curve_max_size = 1000;
		int current_frame_num = 0;

		bool bChebyshevAccOn = false;
		Real rho = Real(0.992);
	};

}