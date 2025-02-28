#ifndef CPP_DOT_PRODUCT_H
#define CPP_DOT_PRODUCT_H
#include "utils.h"

class CppDotProduct : public BaseModel
{
	public:
		CppDotProduct();
		void initialize();
		void execute(
            const std::vector<float*>& dataPointers , const std::vector<std::vector<std::int64_t>>& shapes,
            const std::vector <std::string>& data_types , std::vector<const void*>& outputPointers,
            std::vector<std::vector<std::int64_t>>&  output_shapes) override;
};

#endif