#include "MaterialEditModule.h"
#include "GLVisualModule.h"
//
namespace dyno
{


template<typename Vec3f>
__device__ void RGBtoHSV(const Vec3f& rgb, Vec3f& hsv)
{
	float r = rgb[0] / 255.0f;
	float g = rgb[1] / 255.0f;
	float b = rgb[2] / 255.0f;

	float maxc = fmaxf(r, fmaxf(g, b));
	float minc = fminf(r, fminf(g, b));
	float delta = maxc - minc;

	hsv[2] = maxc; // V

	if (maxc == 0.0f)
	{
		hsv[1] = 0.0f;
		hsv[0] = 0.0f; //
		return;
	}
	else
	{
		hsv[1] = delta / maxc; // S
	}

	if (delta == 0.0f)
	{
		hsv[0] = 0.0f; //
	}
	else if (maxc == r)
	{
		hsv[0] = 60.0f * fmodf(((g - b) / delta), 6.0f);
	}
	else if (maxc == g)
	{
		hsv[0] = 60.0f * (((b - r) / delta) + 2.0f);
	}
	else // maxc == b
	{
		hsv[0] = 60.0f * (((r - g) / delta) + 4.0f);
	}

	if (hsv[0] < 0.0f)
		hsv[0] += 360.0f;
}

__device__ float Overlay(float B, float F)
{
	if (B < 0.5)
		return 2.0 * B * F;
	else
		return 1.0 - 2.0 * (1.0 - B) * (1.0 - F);
}

__device__ float lerp(float a, float b, float t)
{
	return a + t * (b - a);
}

template<typename Vec3f>
__device__ void HSVtoRGB(const Vec3f& hsv, Vec3f& rgb)
{
	float H = hsv[0];
	float S = hsv[1];
	float V = hsv[2];

	float C = V * S;
	float X = C * (1.0f - fabsf(fmodf(H / 60.0f, 2.0f) - 1.0f));
	float m = V - C;

	float r1, g1, b1;

	if (H >= 0 && H < 60)
	{
		r1 = C; g1 = X; b1 = 0;
	}
	else if (H < 120)
	{
		r1 = X; g1 = C; b1 = 0;
	}
	else if (H < 180)
	{
		r1 = 0; g1 = C; b1 = X;
	}
	else if (H < 240)
	{
		r1 = 0; g1 = X; b1 = C;
	}
	else if (H < 300)
	{
		r1 = X; g1 = 0; b1 = C;
	}
	else // H < 360
	{
		r1 = C; g1 = 0; b1 = X;
	}

	rgb[0] = fminf(fmaxf((r1 + m) * 255.0f, 0.0f), 255.0f);
	rgb[1] = fminf(fmaxf((g1 + m) * 255.0f, 0.0f), 255.0f);
	rgb[2] = fminf(fmaxf((b1 + m) * 255.0f, 0.0f), 255.0f);
}

__global__ void colorCorrectionKernel(
	DArray2D<Vec4f> inTex,
	DArray2D<Vec4f> outTex,
	float saturationFactor,
	float hueOffset,
	float contrastFactor,
	float brightness,
	float gamma,
	Vec3f tintColor,      
	float tintIntensity 
)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= inTex.nx() || y >= inTex.ny())
		return;

	Vec4f rgba = inTex(x, y);

	// 0~1 -> 0~255
	Vec3f rgb = Vec3f(rgba.x * 255.0f, rgba.y * 255.0f, rgba.z * 255.0f);

	Vec3f hsv;
	RGBtoHSV(rgb, hsv);

	hsv[0] += hueOffset;
	if (hsv[0] >= 360.0f) hsv[0] -= 360.0f;
	if (hsv[0] < 0.0f) hsv[0] += 360.0f;

	hsv[1] *= saturationFactor;
	hsv[1] = fminf(fmaxf(hsv[1], 0.0f), 1.0f);

	HSVtoRGB(hsv, rgb);
	//contrast
	float fr = (rgb[0] - 128.0f) * contrastFactor + 128.0f;
	float fg = (rgb[1] - 128.0f) * contrastFactor + 128.0f;
	float fb = (rgb[2] - 128.0f) * contrastFactor + 128.0f;
	//brightness
	fr += (brightness - 1) * 255.0f;
	fg += (brightness - 1) * 255.0f;
	fb += (brightness - 1) * 255.0f;
	// Gamma
	fr = 255.0f * powf(fmaxf(fr, 0.0f) / 255.0f, gamma);
	fg = 255.0f * powf(fmaxf(fg, 0.0f) / 255.0f, gamma);
	fb = 255.0f * powf(fmaxf(fb, 0.0f) / 255.0f, gamma);

	fr = fminf(fmaxf(fr, 0.0f), 255.0f) / 255.0f;
	fg = fminf(fmaxf(fg, 0.0f), 255.0f) / 255.0f;
	fb = fminf(fmaxf(fb, 0.0f), 255.0f) / 255.0f;

	fr = Overlay(tintColor[0], fr);
	fg = Overlay(tintColor[1], fg);
	fb = Overlay(tintColor[2], fb);

	outTex(x, y)[0] = fr ;
	outTex(x, y)[1] = fg ;
	outTex(x, y)[2] = fb ;
	outTex(x, y)[3] = rgba.w;
}

__global__ void grayscaleCorrectionKernel(
	DArray2D<float> inTex,
	DArray2D<float> outTex,
	float brightnessOffset,   
	float contrastFactor,     
	float gamma              
)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= inTex.nx() || y >= inTex.ny())
		return;

	float lum = inTex(x, y);  

	float adjusted = lum + (brightnessOffset - 1);

	adjusted = fminf(fmaxf(adjusted, 0.0f), 1.0f);

	adjusted = (adjusted - 0.5f) * contrastFactor + 0.5f;

	adjusted = fminf(fmaxf(adjusted, 0.0f), 1.0f);

	adjusted = powf(adjusted, gamma);

	adjusted = fminf(fmaxf(adjusted, 0.0f), 1.0f);

	outTex(x, y) = adjusted;
}


	IMPLEMENT_CLASS(GrayscaleCorrect)
	GrayscaleCorrect::GrayscaleCorrect()
	{
		initial();
	}

	void GrayscaleCorrect::initial() 
	{
		this->varGamma()->setRange(0, 4);
		this->varContrast()->setRange(0.9, 1.1);
		this->varBrightness()->setRange(0, 2);
		auto fieldChange = std::make_shared<FCallBackFunc>(std::bind(&GrayscaleCorrect::onFieldChanged, this));
		this->inGrayscaleTexture()->attach(fieldChange);

		this->varContrast()->attach(fieldChange);
		this->varGamma()->attach(fieldChange);
		this->varBrightness()->attach(fieldChange);

		this->setName("GrayscaleCorrect");
	}
	GrayscaleCorrect::GrayscaleCorrect(std::shared_ptr<GrayscaleCorrect> other)
	{
		this->varContrast()->setValue(other->varContrast()->getValue());
		this->varGamma()->setValue(other->varGamma()->getValue());
		this->varBrightness()->setValue(other->varBrightness()->getValue());

		other->inGrayscaleTexture()->getSource()->connect(this->inGrayscaleTexture());

		if (!other->outGrayscaleTexture()->isEmpty())
		{
			if (this->outGrayscaleTexture()->isEmpty())
				this->outGrayscaleTexture()->allocate();
			this->outGrayscaleTexture()->assign(other->outGrayscaleTexture()->getData());
		}
		initial();
	}
	void GrayscaleCorrect::onFieldChanged() 
	{
		auto inTexture = this->inGrayscaleTexture()->getData();
		if (this->outGrayscaleTexture()->isEmpty())
			this->outGrayscaleTexture()->allocate();

		auto outTexture = this->outGrayscaleTexture()->getDataPtr();
		auto contrast = this->varContrast()->getValue();
		auto gamma = this->varGamma()->getValue();
		auto brightnessOffset = this->varBrightness()->getValue();
		outTexture->resize(inTexture.nx(), inTexture.ny());

		cuExecute2D(make_uint2(unsigned int(inTexture.nx()), unsigned int(inTexture.ny())),
			grayscaleCorrectionKernel,
			inTexture,
			*outTexture,
			brightnessOffset,
			contrast,
			gamma
		);
	}
	std::shared_ptr<MaterialManagedModule> GrayscaleCorrect::clone() const
	{
		{
			std::shared_ptr<MaterialManagedModule> materialPtr = MaterialManager::getMaterialManagedModule(this->getName());
			if (!materialPtr)
			{
				printf("Error: newGrayscaleCorrect::clone() Failed!! \n");
				return nullptr;
			}

			std::shared_ptr<GrayscaleCorrect> grayCorrect = std::dynamic_pointer_cast<GrayscaleCorrect>(materialPtr);
			if (!grayCorrect)
			{
				printf("Error: newGrayscaleCorrect::clone() Cast Failed!!  \n");
				return nullptr;
			}

			std::shared_ptr<GrayscaleCorrect> newGrayCorrect(new GrayscaleCorrect(grayCorrect));
			newGrayCorrect->setName(MaterialManager::generateUniqueMaterialName(grayCorrect->getName()));
			return newGrayCorrect;
		}
	}

	IMPLEMENT_CLASS(ColorCorrect)
	ColorCorrect::ColorCorrect()
	{
		initial();
	};

	ColorCorrect::ColorCorrect(std::shared_ptr<ColorCorrect> other)
	{
		this->varSaturation()->setValue(other->varSaturation()->getValue());
		this->varHUEOffset()->setValue(other->varHUEOffset()->getValue());
		this->varContrast()->setValue(other->varContrast()->getValue());
		this->varGamma()->setValue(other->varGamma()->getValue());
		this->varTintColor()->setValue(other->varTintColor()->getValue());

		other->inTexture()->getSource()->connect(this->inTexture());

		if (!other->outTexture()->isEmpty())
		{
			if (this->outTexture()->isEmpty())
				this->outTexture()->allocate();
			this->outTexture()->assign(other->outTexture()->getData());
		}
		initial();
	}

	void ColorCorrect::initial() 
	{
		this->varSaturation()->setRange(0, 5);
		this->varGamma()->setRange(0, 4);
		this->varContrast()->setRange(0.9, 1.1);
		this->varHUEOffset()->setRange(0,360);
		this->varTintIntensity()->setRange(0,1);
		this->varBrightness()->setRange(0, 2);

		auto fieldChange = std::make_shared<FCallBackFunc>(std::bind(&ColorCorrect::onFieldChanged, this));
		this->inTexture()->attach(fieldChange);

		this->varSaturation()->attach(fieldChange);
		this->varContrast()->attach(fieldChange);
		this->varGamma()->attach(fieldChange);
		this->varHUEOffset()->attach(fieldChange);
		this->varTintColor()->attach(fieldChange);
		this->varTintIntensity()->attach(fieldChange);
		this->varBrightness()->attach(fieldChange);
		this->inSaturation()->attach(fieldChange);
		this->inHUEOffset()->attach(fieldChange);
		this->inContrast()->attach(fieldChange);
		this->inBrightness()->attach(fieldChange);
		this->inGamma()->attach(fieldChange);
		this->inTintIntensity()->attach(fieldChange);
		this->inTintColor()->attach(fieldChange);

		this->setName("ColorCorrect");

		this->inSaturation()->tagOptional(true);
		this->inHUEOffset()->tagOptional(true);
		this->inContrast()->tagOptional(true);
		this->inBrightness()->tagOptional(true);
		this->inGamma()->tagOptional(true);
		this->inTintIntensity()->tagOptional(true);
		this->inTintColor()->tagOptional(true);

	}

	std::shared_ptr<MaterialManagedModule> ColorCorrect::clone() const
	{
		{
			std::shared_ptr<MaterialManagedModule> materialPtr = MaterialManager::getMaterialManagedModule(this->getName());
			if (!materialPtr)
			{
				printf("Error: newColorCorrect::clone() Failed!! \n");
				return nullptr;
			}

			std::shared_ptr<ColorCorrect> colorCorrect = std::dynamic_pointer_cast<ColorCorrect>(materialPtr);
			if (!colorCorrect)
			{
				printf("Error: newColorCorrect::clone() Cast Failed!!  \n");
				return nullptr;
			}

			std::shared_ptr<ColorCorrect> newColorCorrect(new ColorCorrect(colorCorrect));
			newColorCorrect->setName(MaterialManager::generateUniqueMaterialName(colorCorrect->getName()));
			return newColorCorrect;
		}
	}

	void ColorCorrect::onFieldChanged()
	{
		auto inTexture = this->inTexture()->getData();
		if (this->outTexture()->isEmpty())
			this->outTexture()->allocate();

		auto outTexture = this->outTexture()->getDataPtr();
		auto saturation = this->inSaturation()->isEmpty() ? this->varSaturation()->getValue():this->inSaturation()->getValue();
		auto hueOffset = this->inHUEOffset()->isEmpty() ? this->varHUEOffset()->getValue() : this->inHUEOffset()->getValue();
		auto contrast = this->inContrast()->isEmpty() ? this->varContrast()->getValue(): this->inContrast()->getValue();
		auto gamma = this->inGamma()->isEmpty() ? this->varGamma()->getValue() : this->inGamma()->getValue();
		auto tintColor = this->inTintColor()->isEmpty() ? this->varTintColor()->getValue() : this->inTintColor()->getValue();
		auto brightness = this->inBrightness()->isEmpty() ? this->varBrightness()->getValue() : this->inBrightness()->getValue();

		Vec3f vec3TintColor = Vec3f(tintColor.r,tintColor.g,tintColor.b);
		auto tintIntensity = this->varTintIntensity()->getValue();

		outTexture->resize(inTexture.nx(),inTexture.ny());

		cuExecute2D(make_uint2(unsigned int(inTexture.nx()), unsigned int(inTexture.ny())),
			colorCorrectionKernel,
			inTexture,
			*outTexture,
			saturation,
			hueOffset,
			contrast,
			brightness,
			gamma,
			vec3TintColor,
			tintIntensity
		);
	}


	__global__ void breakTextureKernel(
		DArray2D<Vec4f> inTex,
		DArray2D<float> outR,
		DArray2D<float> outG,
		DArray2D<float> outB,
		DArray2D<float> outA
	) 
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= inTex.nx() || y >= inTex.ny())
			return;

		Vec4f rgba = inTex(x, y);
		outR(x,y) = rgba[0];
		outG(x,y) = rgba[1];
		outB(x,y) = rgba[2];
		outA(x,y) = rgba[3];
	}

	IMPLEMENT_CLASS(BreakTexture)
	void BreakTexture::onFieldChanged()
	{
		if (!this->inTexture()->isEmpty()) 
		{
			auto inTex = this->inTexture()->constDataPtr();

			int x = inTex->nx();
			int y = inTex->ny();


			if (this->outR()->isEmpty())
				this->outR()->allocate();
			if (this->outG()->isEmpty())
				this->outG()->allocate();
			if (this->outB()->isEmpty())
				this->outB()->allocate();
			if (this->outA()->isEmpty())
				this->outA()->allocate();

			this->outR()->resize(inTex->nx(), inTex->ny());
			this->outG()->resize(inTex->nx(), inTex->ny());
			this->outB()->resize(inTex->nx(), inTex->ny());
			this->outA()->resize(inTex->nx(), inTex->ny());

			cuExecute2D(make_uint2(unsigned int(x), unsigned int(y)),
				breakTextureKernel,
				*inTex,
				*this->outR()->getDataPtr(),
				*this->outG()->getDataPtr(),
				*this->outB()->getDataPtr(),
				*this->outA()->getDataPtr()
			);
		}
	}

	BreakTexture::BreakTexture()
	{
		initial();
	}

	BreakTexture::BreakTexture(std::shared_ptr<BreakTexture> other)
	{

		other->inTexture()->getSource()->connect(this->inTexture());

		if (!other->outA()->isEmpty())
		{
			if (this->outA()->isEmpty())
				this->outA()->allocate();
			this->outA()->assign(other->outA()->getData());
		}
		if (!other->outR()->isEmpty())
		{
			if (this->outR()->isEmpty())
				this->outR()->allocate();
			this->outR()->assign(other->outR()->getData());
		}
		if (!other->outG()->isEmpty())
		{
			if (this->outG()->isEmpty())
				this->outG()->allocate();
			this->outG()->assign(other->outG()->getData());
		}
		if (!other->outB()->isEmpty())
		{
			if (this->outB()->isEmpty())
				this->outB()->allocate();
			this->outB()->assign(other->outB()->getData());
		}

		initial();
	}
	std::shared_ptr<MaterialManagedModule> BreakTexture::clone() const
	{
		{
			std::shared_ptr<MaterialManagedModule> materialPtr = MaterialManager::getMaterialManagedModule(this->getName());
			if (!materialPtr)
			{
				printf("Error: BreakTexture::clone() Failed!! \n");
				return nullptr;
			}

			std::shared_ptr<BreakTexture> breakTex = std::dynamic_pointer_cast<BreakTexture>(materialPtr);
			if (!breakTex)
			{
				printf("Error: BreakTexture::clone() Cast Failed!!  \n");
				return nullptr;
			}

			std::shared_ptr<BreakTexture> newBreakTex(new BreakTexture(breakTex));
			newBreakTex->setName(MaterialManager::generateUniqueMaterialName(breakTex->getName()));
			return newBreakTex;
		}
	};
	void BreakTexture::initial()
	{
		auto IndexChange = std::make_shared<FCallBackFunc>(std::bind(&BreakTexture::onFieldChanged, this));
		this->inTexture()->attach(IndexChange);

		this->setName("BreakTexture");
	}

	__device__ float bilinearSample(const float* data, int width, int height, float u, float v) {
		// u,v in [0,1] normalized coordinates
		float x = u * (width - 1);
		float y = v * (height - 1);

		int x0 = floorf(x);
		int y0 = floorf(y);
		int x1 = minimum(x0 + 1, width - 1);
		int y1 = minimum(y0 + 1, height - 1);

		float dx = x - x0;
		float dy = y - y0;

		float v00 = data[y0 * width + x0];
		float v10 = data[y0 * width + x1];
		float v01 = data[y1 * width + x0];
		float v11 = data[y1 * width + x1];

		float v0 = v00 * (1 - dx) + v10 * dx;
		float v1 = v01 * (1 - dx) + v11 * dx;

		return v0 * (1 - dy) + v1 * dy;
	}

	__device__ Vec4f bilinearSample(const Vec4f* data, int width, int height, float u, float v) {
		// u,v in [0,1] normalized coordinates
		float x = u * (width - 1);
		float y = v * (height - 1);

		int x0 = floorf(x);
		int y0 = floorf(y);
		int x1 = minimum(x0 + 1, width - 1);
		int y1 = minimum(y0 + 1, height - 1);

		float dx = x - x0;
		float dy = y - y0;

		Vec4f v00 = data[y0 * width + x0];
		Vec4f v10 = data[y0 * width + x1];
		Vec4f v01 = data[y1 * width + x0];
		Vec4f v11 = data[y1 * width + x1];

		Vec4f v0 = v00 * (1 - dx) + v10 * dx;
		Vec4f v1 = v01 * (1 - dx) + v11 * dx;

		return v0 * (1 - dy) + v1 * dy;
	}

	__global__ void CompositeRGBAKernel(
		DArray2D<float> R,
		DArray2D<float> G,
		DArray2D<float> B,
		DArray2D<float> A,
		DArray2D<Vec4f> outRGBA
	)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= outRGBA.nx() || y >= outRGBA.ny()) return;

		float u = x / float(outRGBA.nx() - 1);
		float v = y / float(outRGBA.ny() - 1);

		float r = bilinearSample(&R(0, 0), R.nx(), R.ny(), u, v);
		float g = bilinearSample(&G(0, 0), G.nx(), G.ny(), u, v);
		float b = bilinearSample(&B(0, 0), B.nx(), B.ny(), u, v);
		float a = bilinearSample(&A(0, 0), A.nx(), A.ny(), u, v);

		outRGBA(x,y) = Vec4f(r, g, b, a);
	}

	bool tryUpdateInFieldValueFromValue(FArray2D<float,GPU>* inField, float value,CArray2D<float>& c) 
	{
		if (!inField->getSource())
		{
			if (inField->isEmpty())
				inField->allocate();
			c.resize(1,1);
			c(0, 0) = value;
			inField->assign(c);
			return true;
		}
	
		return false;
	}

	IMPLEMENT_CLASS(MakeTexture)
	void MakeTexture::onFieldChanged() 
	{
		CArray2D<float> channel;
		tryUpdateInFieldValueFromValue(this->inR(), this->varR()->getValue(),channel);
		tryUpdateInFieldValueFromValue(this->inG(), this->varG()->getValue(),channel);
		tryUpdateInFieldValueFromValue(this->inB(), this->varB()->getValue(),channel);
		tryUpdateInFieldValueFromValue(this->inA(), this->varA()->getValue(),channel);
		channel.clear();

		if (!this->inR()->isEmpty() &&
			!this->inG()->isEmpty() &&
			!this->inB()->isEmpty() &&
			!this->inA()->isEmpty()
			)
		{
			auto inR = this->inR()->constDataPtr();
			int rx = inR->nx();
			int ry = inR->ny();
			auto inG = this->inG()->constDataPtr();
			int gx = inG->nx();
			int gy = inG->ny();
			auto inB = this->inB()->constDataPtr();
			int bx = inB->nx();
			int by = inB->ny();
			auto inA = this->inA()->constDataPtr();
			int ax = inA->nx();
			int ay = inA->ny();

			int x = std::max({ rx, gx, bx, ax });
			int y = std::max({ ry, gy, by, ay });

			if (this->outTexture()->isEmpty())
				this->outTexture()->allocate();
			this->outTexture()->resize(x,y);

			cuExecute2D(make_uint2(unsigned int(x), unsigned int(y)),
				CompositeRGBAKernel,
				this->inR()->getData(),
				this->inG()->getData(),
				this->inB()->getData(),
				this->inA()->getData(),
				*this->outTexture()->getDataPtr()
			);
		}

	}

	MakeTexture::MakeTexture()
	{
		initial();
	}

	MakeTexture::MakeTexture(std::shared_ptr<MakeTexture> other) 
	{
		other->inR()->getSource()->connect(this->inR());
		other->inG()->getSource()->connect(this->inG());
		other->inB()->getSource()->connect(this->inB());
		other->inA()->getSource()->connect(this->inA());

		this->varA()->setValue(other->varA()->getValue());
		this->varR()->setValue(other->varR()->getValue());
		this->varG()->setValue(other->varG()->getValue());
		this->varB()->setValue(other->varB()->getValue());


		if (!other->outTexture()->isEmpty())
		{
			if (this->outTexture()->isEmpty())
				this->outTexture()->allocate();
			this->outTexture()->assign(other->outTexture()->getData());
		}
		initial();
	}
	void MakeTexture::initial() 
	{
		auto fieldChange = std::make_shared<FCallBackFunc>(std::bind(&MakeTexture::onFieldChanged, this));
		this->inR()->attach(fieldChange);
		this->inG()->attach(fieldChange);
		this->inB()->attach(fieldChange);
		this->inA()->attach(fieldChange);
		this->varR()->attach(fieldChange);
		this->varG()->attach(fieldChange);
		this->varB()->attach(fieldChange);
		this->varA()->attach(fieldChange);

		this->varR()->setRange(0, 1);
		this->varG()->setRange(0, 1);
		this->varB()->setRange(0, 1);
		this->varA()->setRange(0, 1);

		this->inR()->tagOptional(true);
		this->inG()->tagOptional(true);
		this->inB()->tagOptional(true);
		this->inA()->tagOptional(true);

		this->setName("MakeTexture");
	}

	std::shared_ptr<MaterialManagedModule> MakeTexture::clone() const 
	{
		std::shared_ptr<MaterialManagedModule> materialPtr = MaterialManager::getMaterialManagedModule(this->getName());
		if (!materialPtr)
		{
			printf("Error: MakeTexture::clone() Failed!! \n");
			return nullptr;
		}

		std::shared_ptr<MakeTexture> makeTex = std::dynamic_pointer_cast<MakeTexture>(materialPtr);
		if (!makeTex)
		{
			printf("Error: MakeTexture::clone() Cast Failed!!  \n");
			return nullptr;
		}

		std::shared_ptr<MakeTexture> newMakeTex(new MakeTexture(makeTex));
		newMakeTex->setName(MaterialManager::generateUniqueMaterialName(newMakeTex->getName()));
		return newMakeTex;
	}

	__global__ void MixRGBA(
		DArray2D<Vec4f> texA,
		DArray2D<Vec4f> texB,
		DArray2D<float> Mask,
		DArray2D<Vec4f> outTex
	)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= outTex.nx() || y >= outTex.ny()) return;

		float u = x / float(outTex.nx() - 1);
		float v = y / float(outTex.ny() - 1);

		Vec4f a = bilinearSample(&texA(0, 0), texA.nx(), texA.ny(), u, v);
		Vec4f b = bilinearSample(&texB(0, 0), texB.nx(), texB.ny(), u, v);
		float w = bilinearSample(&Mask(0, 0), Mask.nx(), Mask.ny(), u, v);

		outTex(x, y) = Vec4f(
			lerp(a[0],b[0],w),
			lerp(a[1],b[1],w),
			lerp(a[2],b[2],w),
			lerp(a[3],b[3],w)
			);
	}

	__global__ void MixFloat(
		DArray2D<float> texA,
		DArray2D<float> texB,
		DArray2D<float> Mask,
		DArray2D<float> outTex
	)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= outTex.nx() || y >= outTex.ny()) return;

		float u = x / float(outTex.nx() - 1);
		float v = y / float(outTex.ny() - 1);

		float a = bilinearSample(&texA(0, 0), texA.nx(), texA.ny(), u, v);
		float b = bilinearSample(&texB(0, 0), texB.nx(), texB.ny(), u, v);
		float w = bilinearSample(&Mask(0, 0), Mask.nx(), Mask.ny(), u, v);

		outTex(x, y) = lerp(a, b, w);
	}

	IMPLEMENT_CLASS(MixTexture)
	MixTexture::MixTexture()
	{
		auto fieldChange = std::make_shared<FCallBackFunc>(std::bind(&MixTexture::onFieldChanged, this));
		this->inTextureA()->attach(fieldChange);
		this->inTextureB()->attach(fieldChange);
		this->inFloatA()->attach(fieldChange);
		this->inFloatB()->attach(fieldChange);

		this->inMask()->attach(fieldChange);
		this->varWeight()->attach(fieldChange);
		this->varWeight()->setRange(0, 1);

		this->inMask()->tagOptional(true);
		this->inTextureA()->tagOptional(true);
		this->inTextureB()->tagOptional(true);
		this->inFloatA()->tagOptional(true);
		this->inFloatB()->tagOptional(true);

	}
	void MixTexture::onFieldChanged() 
	{
		CArray2D<float> C;
		tryUpdateInFieldValueFromValue(this->inMask(), this->varWeight()->getValue(), C);

		if (!this->inTextureA()->isEmpty() && !this->inTextureB()->isEmpty()) {
			int ax = this->inTextureA()->getDataPtr()->nx();
			int ay = this->inTextureA()->getDataPtr()->ny();
			int bx = this->inTextureB()->getDataPtr()->nx();
			int by = this->inTextureB()->getDataPtr()->ny();
			int x = std::max({ ax, bx });
			int y = std::max({ by, ay });
			if (this->outTexture()->isEmpty())
				this->outTexture()->allocate();
			this->outTexture()->resize(x, y);
			cuExecute2D(make_uint2(unsigned int(x), unsigned int(y)),
				MixRGBA,
				this->inTextureA()->getData(),
				this->inTextureB()->getData(),
				this->inMask()->getData(),
				*this->outTexture()->getDataPtr()
			);
		}
		else if (!this->inFloatA()->isEmpty() && !this->inFloatB()->isEmpty()){
			int ax = this->inFloatA()->getDataPtr()->nx();
			int ay = this->inFloatA()->getDataPtr()->ny();
			int bx = this->inFloatB()->getDataPtr()->nx();
			int by = this->inFloatB()->getDataPtr()->ny();
			int x = std::max({ ax, bx });
			int y = std::max({ by, ay });
			if (this->outFloat()->isEmpty())
				this->outFloat()->allocate();
			this->outFloat()->resize(x, y);
			cuExecute2D(make_uint2(unsigned int(x), unsigned int(y)),
				MixFloat,
				this->inFloatA()->getData(),
				this->inFloatB()->getData(),
				this->inMask()->getData(),
				*this->outFloat()->getDataPtr()
			);
		}

	}

	IMPLEMENT_TCLASS(AssignTextureMeshMaterial, TDataType)
	template<typename TDataType>
	AssignTextureMeshMaterial<TDataType>::AssignTextureMeshMaterial()
	{
		auto FieldChange = std::make_shared<FCallBackFunc>(std::bind(&AssignTextureMeshMaterial<TDataType>::updateAssign, this));
		this->varShapeIndex()->attach(FieldChange);
		this->inTextureMesh()->attach(FieldChange);
		this->outTextureMesh()->setDataPtr(std::make_shared<TextureMesh>());
	}

	template<typename TDataType>
	void AssignTextureMeshMaterial<TDataType>::updateAssign()
	{
	
		auto inTexMesh = this->inTextureMesh()->constDataPtr();
		auto inShapes = inTexMesh->shapes();
		auto outTexMesh = this->outTextureMesh()->getDataPtr();

		uint index = this->varShapeIndex()->getValue() < inShapes.size() ? this->varShapeIndex()->getValue() : inShapes.size() - 1;
		

		auto newMat = MaterialManager::getMaterial(this->varMaterialName()->getValue());
		if(newMat != varMat)
			removeMaterialReference();

		varMat = newMat;

		auto inMeshData = inTexMesh->meshDataPtr();
		outTexMesh->meshDataPtr() = inMeshData;
		outTexMesh->shapes() = inTexMesh->shapes();

		auto originalShape = inShapes[index];
		auto replaceShape = std::make_shared<Shape>(*originalShape);
		if (varMat) 
		{
			replaceShape->material = varMat;
		
			auto selfPtr = getSelfPtr();
			if (selfPtr)
				varMat->addAssigner(selfPtr);
			
		}
		

		outTexMesh->shapes()[index] = replaceShape;

		//update visualModule
		auto sinks = this->outTextureMesh()->getSinks();
		for (auto it : sinks)
		{
			GLVisualModule* connectModule = dynamic_cast<GLVisualModule*>(it->parent());
			if (connectModule)
			{
				connectModule->varForceUpdate()->setValue(true);
				connectModule->update();
				connectModule->varForceUpdate()->setValue(false);
			}
		}

	}

	template<typename TDataType>
	void AssignTextureMeshMaterial<TDataType>::compute()
	{
		updateAssign(); 
		
	}

	template<typename TDataType>
	std::shared_ptr<Module> AssignTextureMeshMaterial<TDataType>::getSelfPtr()
	{
		{
			std::shared_ptr<Module> foundModule = nullptr;
			if (this->getParentNode())
			{
				const auto& modules = this->getParentNode()->getModuleList();

				for (const auto& modulePtr : modules) {
					if (modulePtr->objectId() == this->objectId()) {
						foundModule = modulePtr;
						break;
					}
				}

				if (foundModule) {
					std::cout << "found Module£¬ID = " << foundModule->objectId() << std::endl;

				}
				else {
					std::cout << "not found Module" << std::endl;
				}
			}

			return foundModule;
		}
	}

	template<typename TDataType>
	void AssignTextureMeshMaterial<TDataType>::removeMaterialReference()
	{
		{
			auto selfPtr = getSelfPtr();

			if (varMat && selfPtr)
				varMat->removeAssigner(selfPtr);
		}
	}

	DEFINE_CLASS(AssignTextureMeshMaterial);

	IMPLEMENT_CLASS(MatInput)
	void MatInput::onEvent(PKeyboardEvent event)
	{
		if (event.key == PKeyboardType::PKEY_W)
		{
			this->outValue()->setValue(1);
		}
		else if(event.key == PKeyboardType::PKEY_S)
		{
			this->outValue()->setValue(0);
		}

	}

	//IMPLEMENT_TCLASS(TempUpdate, TDataType)
	//template<typename TDataType>
	//void TempUpdate<TDataType>::onFieldChanged()
	//{
	//	if (!matPipeline)
	//		return;

	//	matPipeline->updateMaterialPipline();
	//}
	//DEFINE_CLASS(TempUpdate);

	//	IMPLEMENT_TCLASS(BreakTextureMesh, TDataType)
//
//	template<typename TDataType>
//	BreakTextureMesh<TDataType>::BreakTextureMesh()
//	{
//		this->outMaterialGroup()->setDataPtr(std::make_shared<MaterialGroup>());
//		this->outShapeGroup()->setDataPtr(std::make_shared<ShapeGroup>());
//
//		auto onIndexChangedCallBack = std::make_shared<FCallBackFunc>(std::bind(&BreakTextureMesh<TDataType>::breakTexMesh, this));
//		this->inTextureMesh()->attach(onIndexChangedCallBack);
//
//	}
//
//	template<typename TDataType>
//	BreakTextureMesh<TDataType>::~BreakTextureMesh()
//	{
//
//	}
//
//	template<typename TDataType>
//	void BreakTextureMesh<TDataType>::compute()
//	{
//		breakTexMesh();
//	}
//
//	template<typename TDataType>
//	void BreakTextureMesh<TDataType>::breakTexMesh()
//	{
//		if (this->inTextureMesh()->isEmpty())
//			return;
//
//		auto InTexMesh = this->inTextureMesh()->getDataPtr();
//
//		if (this->outVertices()->isEmpty())
//			this->outVertices()->allocate();
//		if (this->outNormals()->isEmpty())
//			this->outNormals()->allocate();
//		if (this->outTexCoords()->isEmpty())
//			this->outTexCoords()->allocate();
//		if (this->outShapeIds()->isEmpty())
//			this->outShapeIds()->allocate();
//
//		this->outVertices()->getDataPtr()->assign(InTexMesh->meshDataPtr()->vertices());
//		this->outNormals()->getDataPtr()->assign(InTexMesh->meshDataPtr()->normals());
//		this->outTexCoords()->getDataPtr()->assign(InTexMesh->meshDataPtr()->texCoords());
//		this->outShapeIds()->getDataPtr()->assign(InTexMesh->meshDataPtr()->shapeIds());
//
//		auto outMats = this->outMaterialGroup()->getDataPtr();
//		auto outShapes = this->outShapeGroup()->getDataPtr();
//
//		outMats->Materials = std::make_shared<std::vector<std::shared_ptr<Material>>>(InTexMesh->materials());
//		outMats->TexMeshSource = this->inTextureMesh();
//		outShapes->Shapes = std::make_shared<std::vector<std::shared_ptr<Shape>>>(InTexMesh->shapes());
//		outShapes->TexMeshSource = this->inTextureMesh();
//
//	}
//
//	DEFINE_CLASS(BreakTextureMesh);
//
//
//	IMPLEMENT_TCLASS(MakeTextureMesh, TDataType)
//		template<typename TDataType>
//	MakeTextureMesh<TDataType>::MakeTextureMesh()
//	{
//		this->outTextureMesh()->setDataPtr(std::make_shared<TextureMesh>());
//
//		auto changed = std::make_shared<FCallBackFunc>(std::bind(&MakeTextureMesh<TDataType>::onFieldChanged, this));
//		this->inMaterialGroup()->attach(changed);
//		this->inNormals()->attach(changed);
//		this->inShapeGroup()->attach(changed);
//		this->inShapeIds()->attach(changed);
//		this->inTexCoords()->attach(changed);
//		this->inVertices()->attach(changed);
//
//		this->inMaterialGroup()->tagOptional(true);
//	}
//
//	template<typename TDataType>
//	MakeTextureMesh<TDataType>::~MakeTextureMesh()
//	{
//
//	}
//
//	template<typename TDataType>
//	void MakeTextureMesh<TDataType>::compute()
//	{
//
//		onFieldChanged();
//	}
//
//	template<typename TDataType>
//	void MakeTextureMesh<TDataType>::onFieldChanged()
//	{
//		auto outTexMesh = this->outTextureMesh()->getDataPtr();
//
//		if(!this->inVertices()->isEmpty())
//			outTexMesh->meshDataPtr()->vertices().assign(this->inVertices()->getData());
//
//		if (!this->inNormals()->isEmpty())
//			outTexMesh->meshDataPtr()->normals().assign(this->inNormals()->getData());
//
//		if(!this->inTexCoords()->isEmpty())
//			outTexMesh->meshDataPtr()->texCoords().assign(this->inTexCoords()->getData());
//
//		if (!this->inShapeIds()->isEmpty())
//			outTexMesh->meshDataPtr()->shapeIds().assign(this->inShapeIds()->getData());
//
//		if(!this->inMaterialGroup()->isEmpty())
//			outTexMesh->materials() = (*this->inMaterialGroup()->getDataPtr()->Materials);
//
//		if(!this->inShapeGroup()->isEmpty())
//			outTexMesh->shapes() = (*this->inShapeGroup()->getDataPtr()->Shapes);
//
//	}
//
//	DEFINE_CLASS(MakeTextureMesh);
//
//	IMPLEMENT_TCLASS(GetMaterialFromGroup, TDataType)
//
//	template<typename TDataType>
//	GetMaterialFromGroup<TDataType>::GetMaterialFromGroup() 
//	{
//		auto onIndexChangedCallBack = std::make_shared<FCallBackFunc>(std::bind(&GetMaterialFromGroup<TDataType>::onIndexChanged, this));
//
//		this->varIndex()->attach(onIndexChangedCallBack);
//		this->inMaterialGroup()->attach(onIndexChangedCallBack);
//
//		auto onUpdateMaterialCallBack = std::make_shared<FCallBackFunc>(std::bind(&GetMaterialFromGroup<TDataType>::updateMaterial, this));
//
//		this->varAlpha()->attach(onUpdateMaterialCallBack);
//		this->varMetallic()->attach(onUpdateMaterialCallBack);
//		this->varRoughness()->attach(onUpdateMaterialCallBack);
//		this->varColor()->attach(onUpdateMaterialCallBack);
//		this->varBumpScale()->attach(onUpdateMaterialCallBack);
//
//	}
//
//	template<typename TDataType>
//	void GetMaterialFromGroup<TDataType>::compute()
//	{
//		onIndexChanged();
//	}
//	template<typename TDataType>
//	void GetMaterialFromGroup<TDataType>::onIndexChanged()
//	{
//		if (this->inMaterialGroup()->isEmpty())
//			return;
//		if (!this->inMaterialGroup()->constDataPtr())
//			return;
//		auto materials = *this->inMaterialGroup()->constDataPtr()->Materials;
//		uint index = this->varIndex()->getValue();
//
//		if (index >= materials.size())
//		{
//			index = materials.size() - 1;
//		}
//		if (materials.size() > index)
//		{
//			this->mMaterial = materials[index];
//
//			this->varAlpha()->setValue(mMaterial->alpha ,false);
//			this->varMetallic()->setValue(mMaterial->metallic, false);
//			this->varRoughness()->setValue(mMaterial->roughness, false);
//			this->varBumpScale()->setValue(mMaterial->bumpScale, false);
//			this->varColor()->setValue(Color(mMaterial->baseColor.x, mMaterial->baseColor.y, mMaterial->baseColor.z), false);
//			
//			if (this->outTexAlpha()->isEmpty())
//				this->outTexAlpha()->allocate();
//			if (this->outTexBump()->isEmpty())
//				this->outTexBump()->allocate();
//			if (this->outTexColor()->isEmpty())
//				this->outTexColor()->allocate();
//			if (this->outTexORM()->isEmpty())
//				this->outTexORM()->allocate();
//
//			this->outMaterial()->getDataPtr()->alpha = materials[index]->alpha;
//			this->outMaterial()->getDataPtr()->baseColor = materials[index]->baseColor;
//			this->outMaterial()->getDataPtr()->metallic = materials[index]->metallic;
//			this->outMaterial()->getDataPtr()->roughness = materials[index]->roughness;
//			this->outMaterial()->getDataPtr()->bumpScale = materials[index]->bumpScale;
//			this->outMaterial()->getDataPtr()->texAlpha.assign(materials[index]->texAlpha);
//			this->outMaterial()->getDataPtr()->texBump.assign(materials[index]->texBump);
//			this->outMaterial()->getDataPtr()->texColor.assign(materials[index]->texColor);
//			this->outMaterial()->getDataPtr()->texORM.assign(materials[index]->texORM);
//
//
//			this->outTexAlpha()->assign(materials[index]->texAlpha);
//			this->outTexBump()->assign(materials[index]->texBump);
//			this->outTexColor()->assign(materials[index]->texColor);
//			this->outTexORM()->assign(materials[index]->texORM);
//
//		}	
//
//	}
//
//	template<typename TDataType>
//	void GetMaterialFromGroup<TDataType>::updateMaterial()
//	{
//		if (mMaterial) 
//		{
//			mMaterial->alpha = this->varAlpha()->getValue();
//			mMaterial->metallic = this->varMetallic()->getValue();
//			mMaterial->roughness = this->varRoughness()->getValue();
//			mMaterial->bumpScale = this->varBumpScale()->getValue();
//			Color baseColor = this->varColor()->getValue();
//			mMaterial->baseColor = Vec3f(baseColor.r, baseColor.g, baseColor.b);
//			this->inMaterialGroup()->getDataPtr()->TexMeshSource->getDataPtr();
//		}
//	}
//	DEFINE_CLASS(GetMaterialFromGroup);
//
//
//
//	IMPLEMENT_TCLASS(GetShapeFromGroup, TDataType)
//		
//	template<typename TDataType>
//	GetShapeFromGroup<TDataType>::GetShapeFromGroup()
//	{
//		auto changed = std::make_shared<FCallBackFunc>(std::bind(&GetShapeFromGroup<TDataType>::onFieldChanged, this));
//
//		this->varIndex()->attach(changed);
//		this->inShapeGroup()->attach(changed);
//		this->inMaterialOverride()->attach(changed);
//
//		this->inMaterialOverride()->tagOptional(true);
//		this->outShape()->setDataPtr(std::make_shared<Shape>());
//
//	}
//
//	template<typename TDataType>
//	void GetShapeFromGroup<TDataType>::onFieldChanged()
//	{
//		auto inShapes = *this->inShapeGroup()->constDataPtr()->Shapes;
//		int index = this->varIndex()->getValue() >= inShapes.size() ? index = inShapes.size() - 1 : this->varIndex()->getValue();
//
//		auto outShape = this->outShape()->getDataPtr();
//		outShape->boundingBox = inShapes[index]->boundingBox;
//		outShape->boundingTransform = inShapes[index]->boundingTransform;
//		outShape->vertexIndex.assign(inShapes[index]->vertexIndex);
//		outShape->normalIndex.assign(inShapes[index]->normalIndex);
//		outShape->texCoordIndex.assign(inShapes[index]->texCoordIndex);
//
//		if (this->inMaterialOverride()->isEmpty()) 
//		{
//			outShape->material = inShapes[index]->material;
//		}
//		else 
//		{
//			outShape->material = this->inMaterialOverride()->getDataPtr();
//		}
//
//	}
//	DEFINE_CLASS(GetShapeFromGroup);
//
//	IMPLEMENT_TCLASS(MaterialBinder, TDataType)
//	template<typename TDataType>
//	MaterialBinder<TDataType>::MaterialBinder()
//	{
//		auto changed = std::make_shared<FCallBackFunc>(std::bind(&MaterialBinder<TDataType>::onFieldChanged, this));
//
//		this->inShape()->attach(changed);
//		this->inMaterial()->attach(changed);
//
//		this->outShape()->setDataPtr(std::make_shared<Shape>());
//
//	}
//
//	template<typename TDataType>
//	void MaterialBinder<TDataType>::onFieldChanged()
//	{
//		auto inShape = this->inShape()->constDataPtr();
//		auto outShape = this->outShape()->getDataPtr();
//		outShape->boundingBox = inShape->boundingBox;
//		outShape->boundingTransform = inShape->boundingTransform;
//		outShape->vertexIndex.assign(inShape->vertexIndex);
//		outShape->normalIndex.assign(inShape->normalIndex);
//		outShape->texCoordIndex.assign(inShape->texCoordIndex);
//		outShape->material = this->inMaterial()->getDataPtr();
//
//	}
//
//	DEFINE_CLASS(MaterialBinder);
//
//	IMPLEMENT_TCLASS(MakeShapeGroup, TDataType)
//	template<typename TDataType>
//	MakeShapeGroup<TDataType>::MakeShapeGroup()
//	{
//		auto changed = std::make_shared<FCallBackFunc>(std::bind(&MakeShapeGroup<TDataType>::onFieldChanged, this));
//
//		this->inShape0()->attach(changed);
//		this->inShape1()->attach(changed);
//		this->inShape2()->attach(changed);
//		this->inShape3()->attach(changed);
//		this->inShape4()->attach(changed);
//		this->inShape5()->attach(changed);
//		this->inShape6()->attach(changed);
//		this->inShape7()->attach(changed);
//		this->inShape8()->attach(changed);
//		this->inShape9()->attach(changed);
//
//		this->outShapeGroup()->setDataPtr(std::make_shared<ShapeGroup>());
//
//		this->inShape0()->tagOptional(true);
//		this->inShape1()->tagOptional(true);
//		this->inShape2()->tagOptional(true);
//		this->inShape3()->tagOptional(true);
//		this->inShape4()->tagOptional(true);
//		this->inShape5()->tagOptional(true);
//		this->inShape6()->tagOptional(true);
//		this->inShape7()->tagOptional(true);
//		this->inShape8()->tagOptional(true);
//		this->inShape9()->tagOptional(true);
//
//		//ShapeInstances.push_back(this->inShape0());
//		//ShapeInstances.push_back(this->inShape1());
//		//ShapeInstances.push_back(this->inShape2());
//		//ShapeInstances.push_back(this->inShape3());
//		//ShapeInstances.push_back(this->inShape4());
//		//ShapeInstances.push_back(this->inShape5());
//		//ShapeInstances.push_back(this->inShape6());
//		//ShapeInstances.push_back(this->inShape7());
//		//ShapeInstances.push_back(this->inShape8());
//		//ShapeInstances.push_back(this->inShape9());
//
//	}
//
//	template<typename TDataType>
//	void MakeShapeGroup<TDataType>::onFieldChanged()
//	{
//		auto OutShapeGroup = this->outShapeGroup()->getDataPtr();
//		OutShapeGroup->Shapes->clear();
//
//		//for (auto it : ShapeInstances)
//		//{
//		//	if (!it->isEmpty())
//		//	{
//		//		OutShapeGroup->Shapes->push_back(it->getDataPtr());
//		//	}
//		//}
//		if (!this->inShape0()->isEmpty())
//			OutShapeGroup->Shapes->push_back(this->inShape0()->constDataPtr());
//		if (!this->inShape1()->isEmpty())
//			OutShapeGroup->Shapes->push_back(this->inShape1()->constDataPtr());
//		if (!this->inShape2()->isEmpty())
//			OutShapeGroup->Shapes->push_back(this->inShape2()->constDataPtr());
//		if (!this->inShape3()->isEmpty())
//			OutShapeGroup->Shapes->push_back(this->inShape3()->constDataPtr());
//		if (!this->inShape4()->isEmpty())
//			OutShapeGroup->Shapes->push_back(this->inShape4()->constDataPtr());
//		if (!this->inShape5()->isEmpty())
//			OutShapeGroup->Shapes->push_back(this->inShape5()->constDataPtr());
//		if (!this->inShape6()->isEmpty())
//			OutShapeGroup->Shapes->push_back(this->inShape6()->constDataPtr());
//		if (!this->inShape7()->isEmpty())
//			OutShapeGroup->Shapes->push_back(this->inShape7()->constDataPtr());
//		if (!this->inShape8()->isEmpty())
//			OutShapeGroup->Shapes->push_back(this->inShape8()->constDataPtr());
//		if (!this->inShape9()->isEmpty())
//			OutShapeGroup->Shapes->push_back(this->inShape9()->constDataPtr());
//	}
//
//	DEFINE_CLASS(MakeShapeGroup);
//
//	IMPLEMENT_TCLASS(TextureMeshCorrect, TDataType)
//	template<typename TDataType>
//	TextureMeshCorrect<TDataType>::TextureMeshCorrect() 
//	{
//		auto IndexChange = std::make_shared<FCallBackFunc>(std::bind(&TextureMeshCorrect<TDataType>::IndexChange, this));
//		this->varShapeIndex()->attach(IndexChange);
//
//		auto FieldChange = std::make_shared<FCallBackFunc>(std::bind(&TextureMeshCorrect<TDataType>::onFieldChanged, this));
//		this->inAlphaTextureOverride()->attach(FieldChange);
//		this->inColorTextureOverride()->attach(FieldChange);
//		this->inORMTextureOverride()->attach(FieldChange);
//		this->inTextureMesh()->attach(FieldChange);
//		this->varAlpha()->attach(FieldChange);
//		this->varBumpScale()->attach(FieldChange);
//		this->varSaturation()->attach(FieldChange);
//		this->varGamma()->attach(FieldChange);
//		this->varContrast()->attach(FieldChange);
//		this->varOffset()->attach(FieldChange);
//		this->varMetallic()->attach(FieldChange);
//		this->varRoughness()->attach(FieldChange);
//		this->varAO()->attach(FieldChange);
//
//		this->outTextureMesh()->setDataPtr(std::make_shared<TextureMesh>());
//		this->inColorTextureOverride()->tagOptional(true);
//		this->inORMTextureOverride()->tagOptional(true);
//		this->inNormalTextureOverride()->tagOptional(true);
//		this->inAlphaTextureOverride()->tagOptional(true);
//
//	}

	//template<typename TDataType>
	//void TextureMeshCorrect<TDataType>::IndexChange()
	//{
	//}



//	template<typename TDataType>
//	void TextureMeshCorrect<TDataType>::onFieldChanged()
//	{
//		auto outTexMesh = this->outTextureMesh()->getDataPtr();
//		auto inTexMesh = this->inTextureMesh()->constDataPtr();
//
//		outTexMesh->meshData() = inTexMesh->meshData();
//		outTexMesh->shapes() = inTexMesh->shapes();
//
//
//		auto inMaterials = this->inTextureMesh()->constDataPtr()->materials();
//		uint index = this->varShapeIndex()->getValue() < inMaterials.size() ? this->varShapeIndex()->getValue() : inMaterials.size() - 1;
//		
//		auto inMaterial = inMaterials[index];
//
//		std::shared_ptr<Material> outMaterial = MaterialManager::NewMaterial();
//		outMaterial->alpha = inMaterial->alpha;
//		outMaterial->baseColor = inMaterial->baseColor;
//		outMaterial->metallic = inMaterial->metallic;
//		outMaterial->roughness = inMaterial->roughness;
//		outMaterial->bumpScale = inMaterial->bumpScale;
//		outMaterial->texAlpha.assign(inMaterial->texAlpha);
//		outMaterial->texBump.assign(inMaterial->texBump);
//		outMaterial->texColor.assign(inMaterial->texColor);
//		outMaterial->texORM.assign(inMaterial->texORM);
//
//
//		cuExecute(outMaterial->texColor.size(),
//			colorCorrectionKernel,
//			outMaterial->texColor,
//			this->varSaturation()->getValue(),
//			this->varOffset()->getValue(),
//			this->varContrast()->getValue(),
//			this->varGamma()->getValue()
//		);
//	}
//
//	DEFINE_CLASS(TextureMeshCorrect);
//
//
//





//********************************************//
}



