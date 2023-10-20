#include "ImChart.h"

// framework
#include "imgui_extend.h"
#include "imgui.h"

namespace dyno
{
	IMPLEMENT_CLASS(ImChart)

	ImChart::ImChart()
	{
		this->setName("im_colorbar");


		auto callback = std::make_shared<FCallBackFunc>(std::bind(&ImChart::varChange, this));

		this->varInputMode()->attach(callback);	
		this->varCount()->attach(callback);
		this->varFixHeight()->attach(callback);
		this->varMax()->attach(callback);
		this->varMin()->attach(callback);
		this->inFrameNumber()->attach(callback);

		this->varChange();
		this->varCount()->setRange(-50000,50000);
		this->inFrameNumber()->tagOptional(false);

	
	}

	ImChart::~ImChart()
	{
		// mColorBuffer.clear();
	}

	void ImChart::varChange()
	{
		if (this->varInputMode()->getValue()==ImChart::InputMode::Array)
		{
			//this->inValue()->demoteInput();
			this->inArray()->tagOptional(true);
			auto mSinks = this->inArray()->getSinks();
			if (mSinks.size()) 
			{
				for (auto it : mSinks) 
				{
					this->inArray()->disconnect(it);
				}	
			}
		}
		else if(this->varInputMode()->getValue() == ImChart::InputMode::Var)
		{
			//this->inArray()->demoteInput();
			
			this->inValue()->tagOptional(true);
			auto mSinks = this->inValue()->getSinks();
			if (mSinks.size())
			{
				for (auto it : mSinks)
				{
					this->inValue()->disconnect(it);
				}
			}
		}

		
		bool pFixed = this->varFixHeight()->getValue();
		float lowLimit = this->varMin()->getData();
		float upLimit = this->varMax()->getData();

		if (this->varCount()->getValue() >= 1)
			count = this->varCount()->getValue();
	
		
		if (!this->inArray()->isEmpty() && this->varInputMode()->getValue() == InputMode::Array)
		{
			auto pArray = this->inArray()->getData();
			if (pFixed)
			{
				lowLimit = m_reduce_real.minimum(pArray.begin(), pArray.size());
				upLimit = m_reduce_real.maximum(pArray.begin(), pArray.size());
			}

		}
		this->min = lowLimit;
		this->max = upLimit;

		//set data;


		//
		if (this->varInputMode()->getValue() == ImChart::InputMode::Array)
		{
			drawArray();
		}
		else if (this->varInputMode()->getValue() == ImChart::InputMode::Var)
		{
			drawValue();
		}	

		
		if (out)
			this->outputTxt();
		//

	}

	void ImChart::setCoord(ImVec2 coord)
	{
		mCoord = coord;
	}

	ImVec2 ImChart::getCoord() const
	{
		return mCoord;
	}

	bool ImChart::initialize()
	{
		return true;
	}

	void ImChart::update()
	{
		
	}

	void ImChart::paint()
	{
		//std::string labelName = "Label";//+this->varTitle()->getValue()
		char* label = "Label";
		ImGui::PushStyleColor(ImGuiCol_WindowBg, ImGui::ExColorsVal[ImGui::ImGuiExColVal_WindowTopBg_1]);
		ImGui::Begin(label, NULL, /*ImGuiWindowFlags_NoMove |*/  ImGuiWindowFlags_NoTitleBar | /*ImGuiWindowFlags_NoBackground |*/ ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
		ImGui::PopStyleColor();

		std::string checklabel = "Output_" + this->varTitle()->getValue();
		ImGui::Checkbox(checklabel.data(), &out);

		char overlay[2] = " ";
		ImGui::PlotLines(this->varTitle()->getValue().c_str(), valuePtr, count, 0, overlay, min, max, ImVec2(0, 80.0f));
		ImGui::End();

	}

	void ImChart::drawArray()
	{
		if (!this->inArray()->isEmpty())
		{
			d_Value = this->inArray()->getData();
			c_Value.assign(d_Value);
		}

		if (this->varCount()->getValue() >= 1)
		{
			count = this->varCount()->getValue();
			valueVec = c_Value.handle();

			if (valueVec->size() < count)
			{
				int temp = count - valueVec->size();
				for (size_t i = 0; i < temp; i++)
				{
					valueVec->push_back(min);
				}
			}
			//else if (c_Value.size() > count)
			//{
			//	for (size_t i = 0; i < valueVec->size() - count; i++)
			//	{
			//		valueVec->erase(valueVec->end()-1);
			//	}
			//}
		}
		else
		{
			count = c_Value.size();
			//if (count >= 50000)
			//	count = 50000;
		}


		if (c_Value.size())
		{
			valuePtr = &c_Value[0];
		}

	}

	void ImChart::drawValue() 
	{
		if (this->varInputMode()->getValue() == InputMode::Var && !this->inValue()->isEmpty())
		{
			valueVec = c_Value.handle();
			if (this->varCount()->getValue() >= 1 && valueVec->size())
				valueVec->erase(valueVec->begin());
			c_Value.pushBack(this->inValue()->getValue());
		}

		if (this->varCount()->getValue() >= 1)
		{
			count = this->varCount()->getValue();

			if (valueVec->size() < count)
			{
				int temp = count - valueVec->size();
				for (size_t i = 0; i < temp; i++)
				{
					valueVec->push_back(min);
				}
			}
			else if (c_Value.size() > count)
			{
				int temp = valueVec->size();

				for (size_t i = 0; i < temp; i++)
				{
					valueVec->erase(valueVec->end()-1);
				}
			}
		}
		else
		{
			if (c_Value.size())
				count = c_Value.size();
			else
				count = 10;

		}
		

		if (c_Value.size())
		{
			valuePtr = &c_Value[0];
		}


	}

	void ImChart::outputTxt() 
	{

		int out_number = this->inFrameNumber()->getValue();

		std::stringstream ss; ss << out_number;
		std::string filename = this->varOutputPath()->getData() + "_" + this->varTitle()->getValue() + ss.str() + this->file_postfix;// 
		std::ofstream output(filename.c_str(), std::ios::out);

		if (!output.is_open()) 
		{
			printf("------OutputTxt: open file failed \n");
			return;
		}

		std::cout << "------Writer Action!------ " << std::endl;


		output << "x" << "," << "y" << std::endl;

		for (uint i = 0; i < count; i++) {
			output << i+1 << "," << c_Value[i] << std::endl;
		}

		output.close();


		return;
	
	}

}
