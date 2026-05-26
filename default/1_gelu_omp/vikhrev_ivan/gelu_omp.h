#pragma once

#include <vector>
#include <cmath>

std::vector<float> Gelu(const std::vector<float>& input);

std::vector<float> GeluOMP(const std::vector<float>& input);
