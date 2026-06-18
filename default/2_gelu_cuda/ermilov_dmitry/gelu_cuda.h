#pragma once

#include <vector>

std::vector<float> GeluRef(const std::vector<float>& input);
std::vector<float> GeluCUDA(const std::vector<float>& input);
