/**
 * Copyright 2017-2022 Xiaowei He
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "Platform.h"
#include "Matrix.h"

namespace dyno 
{
	enum EnergyType
	{
		Linear,
		StVK,
		NeoHooekean,
		Xuetal,
		MooneyRivlin,
		Fung,
		Ogden,
		Yeoh,
		ArrudaBoyce,
		Fiber
	};

	template<typename Real, typename Matrix>
	class HyperelasticityModel
	{
	public:
		//DYN_FUNC HyperelasticityModel() {};

		DYN_FUNC virtual Real getEnergy(Real lambda1, Real lambda2, Real lambda3) = 0;
		DYN_FUNC virtual Matrix getStressTensorPositive(Real lambda1, Real lambda2, Real lambda3) = 0;
		DYN_FUNC virtual Matrix getStressTensorNegative(Real lambda1, Real lambda2, Real lambda3) = 0;

		Real density;
	};

	//dynamic initialization is not supported for a __constant__ variable
	template<typename Real>
	class LinearModel
	{
	public:
		//dynamic initialization is not supported for a __constant__ variable
		DYN_FUNC LinearModel()
		{
// 			density = Real(1000);
// 			s1 = Real(48000);
// 			s0 = Real(12000);
		}

		DYN_FUNC LinearModel(Real _s0, Real _s1) 
		{
			s0 = _s0;
			s1 = _s1;
			density = Real(1000);
		}

		DYN_FUNC Real getEnergy(Real lambda1, Real lambda2, Real lambda3)
		{
			return s0 * (lambda1 - 1)*(lambda1 - 1) + s0 * (lambda2 - 1)*(lambda2 - 1) + s0 * (lambda3 - 1)*(lambda3 - 1);
		}

		DYN_FUNC SquareMatrix<Real, 3> getStressTensorPositive(Real lambda1, Real lambda2, Real lambda3)
		{
			return s0 * SquareMatrix<Real, 3>::identityMatrix();
		}

		DYN_FUNC SquareMatrix<Real, 3> getStressTensorNegative(Real lambda1, Real lambda2, Real lambda3)
		{
			return s0 * SquareMatrix<Real, 3>::identityMatrix();
		}

		Real s0;
		Real s1;

		Real density;
	};


	template<typename Real>
	class StVKModel
	{
	public:
		//dynamic initialization is not supported for a __constant__ variable
		DYN_FUNC StVKModel()
		{
// 			density = Real(1000);
// 			s1 = Real(48000);
// 			s0 = Real(12000);
		}

		DYN_FUNC StVKModel(Real _s0, Real _s1)
		{
			s0 = _s0;
			s1 = _s1;
			density = Real(1000);
		}

		DYN_FUNC Real getEnergy(Real lambda1, Real lambda2, Real lambda3)
		{
			Real I = lambda1 * lambda1 + lambda2 * lambda2 + lambda3 * lambda3;
			Real sq1 = lambda1 * lambda1;
			Real sq2 = lambda2 * lambda2;
			Real sq3 = lambda3 * lambda3;
			Real II = sq1 * sq1 + sq2 * sq2 + sq3 * sq3;
			return 0.5*s0*(I - 3)*(I - 3) + 0.25*s1*(II - 2 * I + 3);
		}

		DYN_FUNC SquareMatrix<Real, 3> getStressTensorPositive(Real lambda1, Real lambda2, Real lambda3)
		{
			Real I = lambda1 * lambda1 + lambda2 * lambda2 + lambda3 * lambda3;

			Real D1 = 2 * s0*I + s1 * lambda1*lambda1;
			Real D2 = 2 * s0*I + s1 * lambda2*lambda2;
			Real D3 = 2 * s0*I + s1 * lambda3*lambda3;

			SquareMatrix<Real, 3> D;
			D(0, 0) = D1;
			D(1, 1) = D2;
			D(2, 2) = D3;
			return D;
		}

		DYN_FUNC SquareMatrix<Real, 3> getStressTensorNegative(Real lambda1, Real lambda2, Real lambda3)
		{
			SquareMatrix<Real, 3> D = (6 * s0 + s1)*SquareMatrix<Real, 3>::identityMatrix();
			D(0, 0) *= lambda1;
			D(1, 1) *= lambda2;
			D(2, 2) *= lambda3;
			return D;
		}

		Real s0;
		Real s1;

		Real density;
	};

	template<typename Real>
	class NeoHookeanModel
	{
	public:
		//dynamic initialization is not supported for a __constant__ variable
		DYN_FUNC NeoHookeanModel()
		{
// 			s0 = Real(12000);
// 			s1 = Real(48000);
// 			density = Real(1000);
		}

		DYN_FUNC NeoHookeanModel(Real _s0, Real _s1)
		{
			s0 = _s0;
			s1 = _s1;
			density = Real(1000);
		}

		DYN_FUNC Real getEnergy(Real lambda1, Real lambda2, Real lambda3)
		{
			Real I = lambda1 * lambda1 + lambda2 * lambda2 + lambda3 * lambda3;
			Real sqrtIII = lambda1 * lambda2*lambda3;
			return s0 * (I - 3 - 2 * log(sqrtIII)) + s1 * (sqrtIII - 1)*(sqrtIII - 1);
		}

		DYN_FUNC SquareMatrix<Real, 3> getStressTensorPositive(Real lambda1, Real lambda2, Real lambda3)
		{
			Real sq1 = lambda1 * lambda1;
			Real sq2 = lambda2 * lambda2;
			Real sq3 = lambda3 * lambda3;

			Real D1 = 2 * s0 + 2 * s1*sq2*sq3;
			Real D2 = 2 * s0 + 2 * s1*sq3*sq1;
			Real D3 = 2 * s0 + 2 * s1*sq1*sq2;

			SquareMatrix<Real, 3> D;
			D(0, 0) = D1;
			D(1, 1) = D2;
			D(2, 2) = D3;
			return D;
		}

		DYN_FUNC SquareMatrix<Real, 3> getStressTensorNegative(Real lambda1, Real lambda2, Real lambda3)
		{
			Real sqrtIII = lambda1* lambda2* lambda3;

			Real sq1 = lambda1 * lambda1;
			Real sq2 = lambda2 * lambda2;
			Real sq3 = lambda3 * lambda3;

			SquareMatrix<Real, 3> D;
			D(0, 0) = 2 * s0 / sq1 + 2 * s1 * sqrtIII / sq1;
			D(1, 1) = 2 * s0 / sq2 + 2 * s1 * sqrtIII / sq2;
			D(2, 2) = 2 * s0 / sq3 + 2 * s1 * sqrtIII / sq3;

			return D;
		}

		Real s0;
		Real s1;

		Real density;
	};

	template<typename Real, int n>
	class PolynomialModel
	{
	public:
		DYN_FUNC PolynomialModel()
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

		DYN_FUNC Real getEnergy(Real lambda1, Real lambda2, Real lambda3)
		{
			Real sq1 = lambda1 * lambda1;
			Real sq2 = lambda2 * lambda2;
			Real sq3 = lambda3 * lambda3;
			Real I1 = sq1 + sq2 + sq3;
			Real I2 = sq1 * sq2 + sq2 * sq3 + sq3 * sq1;

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

		DYN_FUNC SquareMatrix<Real, 3> getStressTensorPositive(Real lambda1, Real lambda2, Real lambda3)
		{
			Real sq1 = lambda1 * lambda1;
			Real sq2 = lambda2 * lambda2;
			Real sq3 = lambda3 * lambda3;
			Real I1 = sq1 + sq2 + sq3;
			Real I2 = sq1 * sq2 + sq2 * sq3 + sq3 * sq1;

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
						C1_positive = I1 * I2 + 9;
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
						C2_positive = I1 * I2 + 9;
					}

					int exp_i = i < 0 ? 1 : i;
					int exp_j = j < 0 ? 1 : j;
					D1 += 2 * exp_i * C[i][j] * pow(I1 - 3, Real(i_minus_one_even))*pow(I2 - 3, Real(j_even)) * C1_positive + 4 * exp_j * C[i][j] * pow(I1 - 3, Real(i_even))*pow(I2 - 3, Real(j_minus_one_even))*C2_positive*sq1;
					D2 += 2 * exp_i * C[i][j] * pow(I1 - 3, Real(i_minus_one_even))*pow(I2 - 3, Real(j_even)) * C1_positive + 4 * exp_j * C[i][j] * pow(I1 - 3, Real(i_even))*pow(I2 - 3, Real(j_minus_one_even))*C2_positive*sq2;
					D3 += 2 * exp_i * C[i][j] * pow(I1 - 3, Real(i_minus_one_even))*pow(I2 - 3, Real(j_even)) * C1_positive + 4 * exp_j * C[i][j] * pow(I1 - 3, Real(i_even))*pow(I2 - 3, Real(j_minus_one_even))*C2_positive*sq3;

				}
			}

			SquareMatrix<Real, 3> D;
			D(0, 0) = D1 * lambda1;
			D(1, 1) = D2 * lambda2;
			D(2, 2) = D3 * lambda3;

			return D;
		}

		DYN_FUNC SquareMatrix<Real, 3> getStressTensorNegative(Real lambda1, Real lambda2, Real lambda3)
		{
			Real sq1 = lambda1 * lambda1;
			Real sq2 = lambda2 * lambda2;
			Real sq3 = lambda3 * lambda3;
			Real I1 = sq1 + sq2 + sq3;
			Real I2 = sq1 * sq2 + sq2 * sq3 + sq3 * sq1;

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

			SquareMatrix<Real, 3> D;
			D(0, 0) = D1;
			D(1, 1) = D2;
			D(2, 2) = D3;

			return D;
		}

		Real C[n + 1][n + 1];

		Real density;
	};


	template<typename Real>
	class XuModel
	{
	public:
		//dynamic initialization is not supported for a __constant__ variable
		DYN_FUNC XuModel()
		{
// 			density = Real(1000);
// 			s0 = Real(48000);
		}

		DYN_FUNC XuModel(Real _s0)
		{
			s0 = _s0;
			density = Real(1000);
		}

		DYN_FUNC Real getEnergy(Real lambda1, Real lambda2, Real lambda3)
		{
			Real sq1 = lambda1 * lambda1;
			Real sq2 = lambda2 * lambda2;
			Real sq3 = lambda3 * lambda3;
			Real E1 = ((sq1*sq1 - 1) / 4 + (1 / sq1 - 1) / 2) / 3;
			Real E2 = ((sq2*sq2 - 1) / 4 + (1 / sq2 - 1) / 2) / 3;
			Real E3 = ((sq3*sq3 - 1) / 4 + (1 / sq3 - 1) / 2) / 3;

			return s0 * (E1 + E2 + E3);
		}

		DYN_FUNC SquareMatrix<Real, 3> getStressTensorPositive(Real lambda1, Real lambda2, Real lambda3)
		{
			SquareMatrix<Real, 3> D;
			D(0, 0) = s0 * (lambda1*lambda1) / 3;
			D(1, 1) = s0 * (lambda2*lambda2) / 3;
			D(2, 2) = s0 * (lambda3*lambda3) / 3;
			return D;
		}

		DYN_FUNC SquareMatrix<Real, 3> getStressTensorNegative(Real lambda1, Real lambda2, Real lambda3)
		{
			SquareMatrix<Real, 3> D;
			D(0, 0) = lambda1 * s0 / (lambda1*lambda1*lambda1*lambda1) / 3;
			D(1, 1) = lambda2 * s0 / (lambda2*lambda2*lambda2*lambda2) / 3;
			D(2, 2) = lambda3 * s0 / (lambda3*lambda3*lambda3*lambda3) / 3;
			return D;
		}

		Real s0;

		Real density;
	};

	template<typename Real>
	class MooneyRivlinModel
	{
	public:
		DYN_FUNC MooneyRivlinModel()
		{
		}

		DYN_FUNC MooneyRivlinModel(Real _s0, Real _s1, Real _s2)
		{
			s0 = _s0;
			s1 = _s1;
			s2 = _s2;
			density = Real(1000);
		}

		DYN_FUNC Real getEnergy(Real lambda1, Real lambda2, Real lambda3)
		{
			Real I = lambda1 * lambda1 + lambda2 * lambda2 + lambda3 * lambda3;
			Real sq1 = lambda1 * lambda1;
			Real sq2 = lambda2 * lambda2;
			Real sq3 = lambda3 * lambda3;
			Real II = sq1 * sq1 + sq2 * sq2 + sq3 * sq3;
			Real J = lambda1 * lambda2 * lambda3;
			Real sqrtIII = 1 / (lambda1 * lambda2 * lambda3);
			Real cbrtIII = 1 / cbrt(sq1 * sq2 * sq3);
			return s0 * (cbrtIII * I - 3) + s1 * (J - 1) * (J - 1) + s2 * (0.5 * cbrtIII * cbrtIII * (I * I - II) - 3);
		}

		DYN_FUNC SquareMatrix<Real, 3> getStressTensorPositive(Real lambda1, Real lambda2, Real lambda3)
		{
			Real sq1 = lambda1 * lambda1;
			Real sq2 = lambda2 * lambda2;
			Real sq3 = lambda3 * lambda3;

			Real I = lambda1 * lambda1 + lambda2 * lambda2 + lambda3 * lambda3;
			Real II = sq1 * sq1 + sq2 * sq2 + sq3 * sq3;
			Real J = lambda1 * lambda2 * lambda3;
			Real sqrtIII = 1 / (lambda1 * lambda2 * lambda3);
			Real cbrtIII = 1 / cbrt(sq1 * sq2 * sq3);

			Real D1 = 2 * s0 * cbrtIII + 2.0 * s1 * J * J + 2.0 / 3.0 * s2 / lambda1 * II * cbrtIII * cbrtIII + 2 * s2 * I * cbrtIII * cbrtIII;
			Real D2 = 2 * s0 * cbrtIII + 2.0 * s1 * J * J + 2.0 / 3.0 * s2 / lambda2 * II * cbrtIII * cbrtIII + 2 * s2 * I * cbrtIII * cbrtIII;
			Real D3 = 2 * s0 * cbrtIII + 2.0 * s1 * J * J + 2.0 / 3.0 * s2 / lambda3 * II * cbrtIII * cbrtIII + 2 * s2 * I * cbrtIII * cbrtIII;

			SquareMatrix<Real, 3> D;
			D(0, 0) = D1;
			D(1, 1) = D2;
			D(2, 2) = D3;
			return D;
		}

		DYN_FUNC SquareMatrix<Real, 3> getStressTensorNegative(Real lambda1, Real lambda2, Real lambda3)
		{
			Real sq1 = lambda1 * lambda1;
			Real sq2 = lambda2 * lambda2;
			Real sq3 = lambda3 * lambda3;

			Real I = lambda1 * lambda1 + lambda2 * lambda2 + lambda3 * lambda3;
			Real II = sq1 * sq1 + sq2 * sq2 + sq3 * sq3;
			Real J = lambda1 * lambda2 * lambda3;
			Real sqrtIII = 1 / (lambda1 * lambda2 * lambda3);
			Real cbrtIII = 1 / cbrt(sq1 * sq2 * sq3);

			SquareMatrix<Real, 3> D;
			D(0, 0) = 2.0 / 3.0 * s0 / lambda1 * cbrtIII * I + 2 * s1 / lambda1 * J + 2.0 / 3.0 * s2 * cbrtIII * cbrtIII / lambda1 * I * I + 2 * s2 * lambda1 * lambda1 * lambda1 * cbrtIII * cbrtIII;
			D(1, 1) = 2.0 / 3.0 * s0 / lambda2 * cbrtIII * I + 2 * s1 / lambda2 * J + 2.0 / 3.0 * s2 * cbrtIII * cbrtIII / lambda2 * I * I + 2 * s2 * lambda2 * lambda2 * lambda2 * cbrtIII * cbrtIII;
			D(2, 2) = 2.0 / 3.0 * s0 / lambda3 * cbrtIII * I + 2 * s1 / lambda3 * J + 2.0 / 3.0 * s2 * cbrtIII * cbrtIII / lambda3 * I * I + 2 * s2 * lambda3 * lambda3 * lambda3 * cbrtIII * cbrtIII;

			return D;
		}

		Real s0;
		Real s1;
		Real s2;
		Real density;
	};

	template<typename Real>
	class FungModel
	{
	public:
		DYN_FUNC FungModel()
		{
		}

		DYN_FUNC FungModel(Real _s0, Real _s1, Real _s2, Real _s3)
		{
			s0 = _s0;
			s1 = _s1;
			s2 = _s2;
			s3 = _s3;
			density = Real(1000);
		}

		DYN_FUNC Real getEnergy(Real lambda1, Real lambda2, Real lambda3)
		{
			Real I = lambda1 * lambda1 + lambda2 * lambda2 + lambda3 * lambda3;
			Real sq1 = lambda1 * lambda1;
			Real sq2 = lambda2 * lambda2;
			Real sq3 = lambda3 * lambda3;
			Real II = sq1 * sq1 + sq2 * sq2 + sq3 * sq3;
			Real J = lambda1 * lambda2 * lambda3;
			Real sqrtIII = 1 / (lambda1 * lambda2 * lambda3);
			Real cbrtIII = 1 / cbrt(sq1 * sq2 * sq3);
			return s0 * (cbrtIII * I - 3) + s1 * (J - 1) * (J - 1) + s2 * exp(s3 * (cbrtIII * I - 3) - 1);
		}

		DYN_FUNC SquareMatrix<Real, 3> getStressTensorPositive(Real lambda1, Real lambda2, Real lambda3)
		{
			Real sq1 = lambda1 * lambda1;
			Real sq2 = lambda2 * lambda2;
			Real sq3 = lambda3 * lambda3;

			Real I = lambda1 * lambda1 + lambda2 * lambda2 + lambda3 * lambda3;
			Real II = sq1 * sq1 + sq2 * sq2 + sq3 * sq3;
			Real J = lambda1 * lambda2 * lambda3;
			Real sqrtIII = 1 / (lambda1 * lambda2 * lambda3);
			Real cbrtIII = 1 / cbrt(sq1 * sq2 * sq3);

			Real D1 = 2 * s0 * cbrtIII + 2.0 * s1 * J * J + 2 * s2 * s3 * cbrtIII * exp(s3 * (cbrtIII * I - 3));
			Real D2 = 2 * s0 * cbrtIII + 2.0 * s1 * J * J + 2 * s2 * s3 * cbrtIII * exp(s3 * (cbrtIII * I - 3));
			Real D3 = 2 * s0 * cbrtIII + 2.0 * s1 * J * J + 2 * s2 * s3 * cbrtIII * exp(s3 * (cbrtIII * I - 3));

			SquareMatrix<Real, 3> D;
			D(0, 0) = D1;
			D(1, 1) = D2;
			D(2, 2) = D3;
			return D;
		}

		DYN_FUNC SquareMatrix<Real, 3> getStressTensorNegative(Real lambda1, Real lambda2, Real lambda3)
		{
			Real sq1 = lambda1 * lambda1;
			Real sq2 = lambda2 * lambda2;
			Real sq3 = lambda3 * lambda3;

			Real I = lambda1 * lambda1 + lambda2 * lambda2 + lambda3 * lambda3;
			Real II = sq1 * sq1 + sq2 * sq2 + sq3 * sq3;
			Real J = lambda1 * lambda2 * lambda3;
			Real sqrtIII = 1 / (lambda1 * lambda2 * lambda3);
			Real cbrtIII = 1 / cbrt(sq1 * sq2 * sq3);

			SquareMatrix<Real, 3> D;
			D(0, 0) = 2.0 / 3.0 * s0 / lambda1 * cbrtIII * I + 2 * s1 / lambda1 * J + 2.0 / 3.0 * s2 * s3 * cbrtIII / lambda1 * I * exp(s3 * (cbrtIII * I - 3));
			D(1, 1) = 2.0 / 3.0 * s0 / lambda2 * cbrtIII * I + 2 * s1 / lambda2 * J + 2.0 / 3.0 * s2 * s3 * cbrtIII / lambda2 * I * exp(s3 * (cbrtIII * I - 3));
			D(2, 2) = 2.0 / 3.0 * s0 / lambda3 * cbrtIII * I + 2 * s1 / lambda2 * J + 2.0 / 3.0 * s2 * s3 * cbrtIII / lambda3 * I * exp(s3 * (cbrtIII * I - 3));

			return D;
		}

		Real s0;
		Real s1;
		Real s2;
		Real s3;
		Real density;
	};

	template<typename Real>
	class ArrudaBoyceModel
	{
	public:
		DYN_FUNC ArrudaBoyceModel()
		{
		}

		DYN_FUNC ArrudaBoyceModel(Real _s0, Real _s1, Real _s2)
		{
			s0 = _s0;
			s1 = _s1;
			s2 = _s2;
			density = Real(1000);
		}

		DYN_FUNC Real getEnergy(Real lambda1, Real lambda2, Real lambda3)
		{
			Real I = lambda1 * lambda1 + lambda2 * lambda2 + lambda3 * lambda3;
			Real sq1 = lambda1 * lambda1;
			Real sq2 = lambda2 * lambda2;
			Real sq3 = lambda3 * lambda3;
			Real II = sq1 * sq1 + sq2 * sq2 + sq3 * sq3;
			//Real J = lambda1 * lambda2 * lambda3;
			//Real sqrtIII = 1 / (lambda1 * lambda2 * lambda3);
			//Real cbrtIII = 1 / cbrt(sq1 * sq2 * sq3);
			return 0.5 * s0 * (II - 3) + 0.25 * s0 * s1 * (II * II - 9) + s0 * s2 / 6.0 * (II * II * II - 27);
		}

		DYN_FUNC SquareMatrix<Real, 3> getStressTensorPositive(Real lambda1, Real lambda2, Real lambda3)
		{
			Real sq1 = lambda1 * lambda1;
			Real sq2 = lambda2 * lambda2;
			Real sq3 = lambda3 * lambda3;

			Real II = sq1 * sq1 + sq2 * sq2 + sq3 * sq3;

			Real D1 = 2 * s0 * lambda1 * lambda1 + 2.0 * lambda1 * lambda1 * s0 * s1 * II + 2.0 * s0 * s2 * lambda1 * lambda1 * II * II;
			Real D2 = 2 * s0 * lambda2 * lambda2 + 2.0 * lambda2 * lambda2 * s0 * s1 * II + 2.0 * s0 * s2 * lambda2 * lambda2 * II * II;
			Real D3 = 2 * s0 * lambda3 * lambda3 + 2.0 * lambda3 * lambda3 * s0 * s1 * II + 2.0 * s0 * s2 * lambda3 * lambda3 * II * II;

			SquareMatrix<Real, 3> D;
			D(0, 0) = D1;
			D(1, 1) = D2;
			D(2, 2) = D3;
			return D;
		}

		DYN_FUNC SquareMatrix<Real, 3> getStressTensorNegative(Real lambda1, Real lambda2, Real lambda3)
		{

			SquareMatrix<Real, 3> D;
			D(0, 0) = 0.;
			D(1, 1) = 0.;
			D(2, 2) = 0.;

			return D;
		}

		Real s0;
		Real s1;
		Real s2;
		Real density;
	};

	template<typename Real>
	class OgdenModel
	{
	public:
		DYN_FUNC OgdenModel()
		{
		}

		DYN_FUNC OgdenModel(Real _s0, Real _s1)
		{
			s0 = _s0;
			s1 = _s1;
			density = Real(1000);
		}

		DYN_FUNC Real getEnergy(Real lambda1, Real lambda2, Real lambda3)
		{
			Real I = lambda1 * lambda1 + lambda2 * lambda2 + lambda3 * lambda3;
			Real sq1 = lambda1 * lambda1;
			Real sq2 = lambda2 * lambda2;
			Real sq3 = lambda3 * lambda3;
			Real II = sq1 * sq1 + sq2 * sq2 + sq3 * sq3;
			Real J = lambda1 * lambda2 * lambda3;
			Real sqrtIII = 1 / (lambda1 * lambda2 * lambda3);
			Real cbrtIII = 1 / cbrt(sq1 * sq2 * sq3);
			return s0 * (II - 3) + 2.0 * s0 * log(J) + s1 * (J - 1) * (J - 1);
		}

		DYN_FUNC SquareMatrix<Real, 3> getStressTensorPositive(Real lambda1, Real lambda2, Real lambda3)
		{
			Real sq1 = lambda1 * lambda1;
			Real sq2 = lambda2 * lambda2;
			Real sq3 = lambda3 * lambda3;

			Real II = sq1 * sq1 + sq2 * sq2 + sq3 * sq3;
			Real J = lambda1 * lambda2 * lambda3;
			Real sqrtIII = 1 / (lambda1 * lambda2 * lambda3);
			Real cbrtIII = 1 / cbrt(sq1 * sq2 * sq3);

			Real D1 = 4.0 * s0 * lambda1 * lambda1 * lambda1 + 2.0 * s1 * J * J / lambda1;
			Real D2 = 4.0 * s0 * lambda2 * lambda2 * lambda2 + 2.0 * s1 * J * J / lambda2;
			Real D3 = 4.0 * s0 * lambda3 * lambda3 * lambda3 + 2.0 * s1 * J * J / lambda3;

			SquareMatrix<Real, 3> D;
			D(0, 0) = D1 / lambda1;
			D(1, 1) = D2 / lambda2;
			D(2, 2) = D3 / lambda3;
			return D;
		}

		DYN_FUNC SquareMatrix<Real, 3> getStressTensorNegative(Real lambda1, Real lambda2, Real lambda3)
		{
			Real sq1 = lambda1 * lambda1;
			Real sq2 = lambda2 * lambda2;
			Real sq3 = lambda3 * lambda3;

			Real II = sq1 * sq1 + sq2 * sq2 + sq3 * sq3;
			Real J = lambda1 * lambda2 * lambda3;
			Real sqrtIII = 1 / (lambda1 * lambda2 * lambda3);
			Real cbrtIII = 1 / cbrt(sq1 * sq2 * sq3);
			SquareMatrix<Real, 3> D;
			D(0, 0) = 4.0 * s0 / lambda1 + 2.0 * s1 * J / lambda1;
			D(1, 1) = 4.0 * s0 / lambda2 + 2.0 * s1 * J / lambda2;
			D(2, 2) = 4.0 * s0 / lambda3 + 2.0 * s1 * J / lambda3;

			return D;
		}

		Real s0;
		Real s1;
		Real density;
	};

	template<typename Real>
	class YeohModel
	{
	public:
		DYN_FUNC YeohModel()
		{
		}

		DYN_FUNC YeohModel(Real _s0, Real _s1, Real _s2)
		{
			s0 = _s0;
			s1 = _s1;
			s2 = _s2;
			density = Real(1000);
		}

		DYN_FUNC Real getEnergy(Real lambda1, Real lambda2, Real lambda3)
		{
			Real I = lambda1 * lambda1 + lambda2 * lambda2 + lambda3 * lambda3;
			return s0 * (I - 3) + s1 * (I - 3) * (I - 3)/* + s2 * (I - 3) * (I - 3) * (I - 3)*/;
		}

		DYN_FUNC SquareMatrix<Real, 3> getStressTensorPositive(Real lambda1, Real lambda2, Real lambda3)
		{


			Real I = lambda1 * lambda1 + lambda2 * lambda2 + lambda3 * lambda3;


			Real D1 = 2.0 * s0 + 4 * s1 * I + 6 * lambda1 * s2 * I * I + 54 * s2;
			Real D2 = 2.0 * s0 + 4 * s1 * I + 6 * lambda2 * s2 * I * I + 54 * s2;
			Real D3 = 2.0 * s0 + 4 * s1 * I + 6 * lambda3 * s2 * I * I + 54 * s2;

			SquareMatrix<Real, 3> D;
			D(0, 0) = D1;
			D(1, 1) = D2;
			D(2, 2) = D3;
			return D;
		}

		DYN_FUNC SquareMatrix<Real, 3> getStressTensorNegative(Real lambda1, Real lambda2, Real lambda3)
		{


			Real I = lambda1 * lambda1 + lambda2 * lambda2 + lambda3 * lambda3;

			SquareMatrix<Real, 3> D;
			D(0, 0) = 12.0 * lambda1 * s1 + 36.0 * lambda1 * s2 * I;
			D(1, 1) = 12.0 * lambda2 * s1 + 36.0 * lambda2 * s2 * I;
			D(2, 2) = 12.0 * lambda3 * s1 + 36.0 * lambda3 * s2 * I;

			return D;
		}

		Real s0;
		Real s1;
		Real s2;
		Real density;
	};

	//dynamic initialization is not supported for a __constant__ variable
	template<typename Real>
	class FiberModel
	{
	public:
		//dynamic initialization is not supported for a __constant__ variable
		DYN_FUNC FiberModel()
		{
			// 			density = Real(1000);
		/*	s[0] = 1;
			s[1] = 1;
			s[2] = 1;
			*/
		}

		DYN_FUNC FiberModel(Real _s0, Real _s1, Real _s2)
		{
			s[0] = _s0;
			s[1] = _s1;
			s[2] = _s2;
			density = Real(1);
		}

		DYN_FUNC inline SquareMatrix<Real, 3> getF_TF(Real lambda1, Real lambda2, Real lambda3, SquareMatrix<Real, 3> V) {
			auto res = SquareMatrix<Real, 3>::identityMatrix();
			res(0, 0) = lambda1 * lambda1;
			res(1, 1) = lambda2 * lambda2;
			res(2, 2) = lambda3 * lambda3;
			res = V * res * V.transpose();
			return res;
			
		}
		
		DYN_FUNC inline SquareMatrix<Real, 3> get_diag(Real l1, Real l2, Real l3) {
			auto res = SquareMatrix<Real, 3>::identityMatrix();
			res(0, 0) = l1;
			res(1, 1) = l2;
			res(2, 2) = l3;
			return res;
		}
		DYN_FUNC inline SquareMatrix<Real, 3> get_diag_bool(bool l1, bool l2, bool l3) {
			auto res = SquareMatrix<Real, 3>::identityMatrix();
			res(0, 0) = l1 ? (Real) 1 : (Real) 0;
			res(1, 1) = l2 ? (Real) 1 : (Real) 0;
			res(2, 2) = l3 ? (Real) 1 : (Real) 0;
			return res;
			
		}
		DYN_FUNC inline Real getJ(Real l1, Real l2, Real l3) {
			return l1 * l2 * l3;
		}
		DYN_FUNC inline SquareMatrix<Real, 3> get_entry(int i, int j) {
			auto res = 0 * SquareMatrix<Real, 3>::identityMatrix();
			if (i >= 3 || j >= 3 || i < 0 || j < 0)
				return res;
			res(i, j) = (Real) 1.0;
			res(j, i) = (Real) 1.0;
			return res;
		}

		DYN_FUNC Real getEnergy(Real lambda1, Real lambda2, Real lambda3, SquareMatrix<Real,3> V)
		{
			auto FTF = getF_TF(lambda1, lambda2, lambda3, V);
			Real energy = 0.0;
			for (int i = 0; i < 3; i++) {
				energy += pow(s[i] * (FTF(i, i) - 1), 2);
				energy += s[i] * s[(i + 1) % 3] * pow(FTF(i, (i + 1) % 3), 2) / (FTF(i, i) * FTF((i + 1) % 3, (i + 1) % 3));
			}
			energy += s[0] * s[1] * s[2] * pow(getJ(lambda1, lambda2, lambda3) -1,2);
			return density * energy;
		}

		DYN_FUNC SquareMatrix<Real, 3> getStressTensorPositive(Real lambda1, Real lambda2, Real lambda3, SquareMatrix<Real, 3> V)
		{
			auto res = 0 * SquareMatrix<Real, 3>::identityMatrix();
			auto FTF = getF_TF(lambda1, lambda2, lambda3, V);
			auto J = getJ(lambda1, lambda2, lambda3);
			for (int i = 0; i < 3; i++) {
				if (FTF(i, i) >= (1.0+0.05)) {
					res += 400 * (FTF(i, i) - 1) * pow(s[i], 2) * get_diag_bool(i==0, i==1, i==2);
					}
				/*
				if (FTF(i, (i + 1) % 3) >= 0.05)
					res += 2 * s[i] * s[(i + 1) % 3] * FTF(i, (i + 1) % 3) / (FTF(i, i) * FTF((i + 1) % 3, (i + 1) % 3)) * get_entry(i, (i + 1) % 3); */
			}
			res = get_diag(lambda1, lambda2, lambda3) * V.transpose() * res * V * get_diag(1 / lambda1, 1 / lambda2, 1 / lambda3);
			if (J >= (1.0 + 0.05)) {
				res += 20 * J * (J-1) * s[0] * s[1] * s[2] * get_diag(1/pow(lambda1,2), 1/pow(lambda2,2), 1/pow(lambda3,2));
			}
			
			return density * res;
		}

		DYN_FUNC SquareMatrix<Real, 3> getStressTensorNegative(Real lambda1, Real lambda2, Real lambda3, SquareMatrix<Real, 3> V)
		{
			auto res = 0 *  SquareMatrix<Real, 3>::identityMatrix();
			auto FTF = getF_TF(lambda1, lambda2, lambda3, V);
			auto J = getJ(lambda1, lambda2, lambda3);

			for (int i = 0; i < 3; i++) {
				if (FTF(i, i) < (1.0 -0.05) )
				{
					res += 400 * (1 - FTF(i, i)) * pow(s[i], 2) * get_diag_bool(i == 0, i == 1, i == 2);
				}
				
				/*if (FTF(i, (i + 1) % 3) <= -0.05)*/
					res += 200 * s[i] * s[(i + 1) % 3] * FTF(i, (i + 1) % 3) / (FTF(i, i) * FTF((i + 1) % 3, (i + 1) % 3)) * get_entry(i, (i + 1) % 3);
				res += 200 * s[i] * s[(i + 1) % 3] * pow(FTF(i, (i + 1) % 3),2) / (FTF(i, i) * FTF((i + 1) % 3, (i + 1) % 3)) * (
					get_diag_bool(i==0, i==1, i==2) / FTF(i, i) +
					get_diag_bool((i+1)%3 == 0, (i+1)%3 == 1, (i+1)%3 == 2) / FTF((i + 1) % 3, (i + 1) % 3)     );
			}
			
			res = get_diag(lambda1, lambda2, lambda3) * V.transpose() * res * V;
			
			if (J < (1.0 - 0.05)) {
				res += 20 * J * (1 - J) * s[0] * s[1] * s[2] * get_diag(1/lambda1, 1/lambda2, 1/lambda3);
			}
			
			return density * res;
		}
		DYN_FUNC void getInfo(Real lambda1, Real lambda2, Real lambda3, SquareMatrix<Real, 3> V)
		{
			/*
			auto FTF = getF_TF(lambda1, lambda2, lambda3, V);
			printf("=========================FTF=========================\nrow1:%f,\t%f,\t%f\nrow2:%f,\t%f,\t%f\nrow3:%f,\t%f,\t%f\n", FTF(0, 0), FTF(0, 1), FTF(0, 2),FTF(1, 0), FTF(1, 1), FTF(1, 2), FTF(2, 0), FTF(2, 1), FTF(2, 2));
			*/
		}
		Real s[3];

		Real density;
	};

	template<typename Real>
	struct EnergyModels
	{
		LinearModel<Real> linearModel;
		StVKModel<Real> stvkModel;
		NeoHookeanModel<Real> neohookeanModel;
		XuModel<Real> xuModel;
		MooneyRivlinModel<Real> mrModel;
		FungModel<Real> fungModel;
		OgdenModel<Real> ogdenModel;
		YeohModel<Real> yeohModel;
		ArrudaBoyceModel<Real> abModel;
		FiberModel<Real> fiberModel;
	};
}