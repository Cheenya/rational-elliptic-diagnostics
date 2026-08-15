#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using Int128 = __int128;
using UInt128 = unsigned __int128;

std::uint64_t absolute_value(std::int64_t value) {
    if (value >= 0) {
        return static_cast<std::uint64_t>(value);
    }
    return static_cast<std::uint64_t>(-(value + 1)) + 1;
}

std::vector<std::pair<std::uint64_t, int>> factorize(std::uint64_t value) {
    std::vector<std::pair<std::uint64_t, int>> factors;
    for (std::uint64_t divisor = 2; divisor <= value / divisor; ++divisor) {
        if (value % divisor != 0) {
            continue;
        }
        int exponent = 0;
        do {
            value /= divisor;
            ++exponent;
        } while (value % divisor == 0);
        factors.push_back({divisor, exponent});
    }
    if (value > 1) {
        factors.push_back({value, 1});
    }
    return factors;
}

std::vector<std::uint64_t> positive_divisors(
    const std::vector<std::pair<std::uint64_t, int>>& factors,
    int exponent_multiplier
) {
    std::vector<std::uint64_t> divisors = {1};
    for (const auto& [prime, exponent] : factors) {
        const std::vector<std::uint64_t> previous = divisors;
        std::uint64_t prime_power = 1;
        for (int power = 1; power <= exponent * exponent_multiplier; ++power) {
            if (prime_power > std::numeric_limits<std::uint64_t>::max() / prime) {
                break;
            }
            prime_power *= prime;
            for (std::uint64_t divisor : previous) {
                if (divisor <= std::numeric_limits<std::uint64_t>::max() / prime_power) {
                    divisors.push_back(divisor * prime_power);
                }
            }
        }
    }
    std::sort(divisors.begin(), divisors.end());
    divisors.erase(std::unique(divisors.begin(), divisors.end()), divisors.end());
    return divisors;
}

std::vector<std::int64_t> signed_divisors(
    const std::vector<std::pair<std::uint64_t, int>>& factors,
    int exponent_multiplier
) {
    std::vector<std::int64_t> candidates;
    for (std::uint64_t divisor : positive_divisors(factors, exponent_multiplier)) {
        if (divisor > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            continue;
        }
        const auto value = static_cast<std::int64_t>(divisor);
        candidates.push_back(-value);
        candidates.push_back(value);
    }
    return candidates;
}

Int128 cubic(std::int64_t a, std::int64_t b, std::int64_t x) {
    const Int128 value = x;
    return value * value * value + static_cast<Int128>(a) * value + b;
}

Int128 third_division_polynomial(std::int64_t a, std::int64_t b, std::int64_t x) {
    const Int128 value = x;
    const Int128 coefficient_a = a;
    return 3 * value * value * value * value
        + 6 * coefficient_a * value * value
        + 12 * static_cast<Int128>(b) * value
        - coefficient_a * coefficient_a;
}

bool is_positive_square(Int128 value) {
    if (value <= 0) {
        return false;
    }
    const auto target = static_cast<UInt128>(value);
    std::uint64_t root = static_cast<std::uint64_t>(
        std::sqrt(static_cast<long double>(value))
    );
    const auto square = [](std::uint64_t candidate) -> UInt128 {
        return static_cast<UInt128>(candidate) * candidate;
    };
    while (square(root) > target) {
        --root;
    }
    while (root < std::numeric_limits<std::uint64_t>::max() && square(root + 1) <= target) {
        ++root;
    }
    return square(root) == target;
}

std::optional<std::int64_t> exact_integer_cube_root(Int128 value) {
    const bool negative = value < 0;
    const UInt128 target = negative
        ? static_cast<UInt128>(-value)
        : static_cast<UInt128>(value);
    std::uint64_t low = 0;
    std::uint64_t high = 1;
    const auto cube = [](std::uint64_t candidate) -> UInt128 {
        return static_cast<UInt128>(candidate) * candidate * candidate;
    };
    while (cube(high) < target) {
        high *= 2;
    }
    while (low <= high) {
        const std::uint64_t middle = low + (high - low) / 2;
        const UInt128 middle_cube = cube(middle);
        if (middle_cube == target) {
            const auto root = static_cast<std::int64_t>(middle);
            return negative ? -root : root;
        }
        if (middle_cube < target) {
            low = middle + 1;
        } else {
            if (middle == 0) {
                break;
            }
            high = middle - 1;
        }
    }
    return std::nullopt;
}

std::vector<std::int64_t> two_torsion_roots(std::int64_t a, std::int64_t b) {
    std::vector<std::int64_t> candidates;
    if (b == 0) {
        candidates.push_back(0);
        if (a != 0) {
            candidates = signed_divisors(factorize(absolute_value(a)), 1);
            candidates.push_back(0);
        }
    } else {
        candidates = signed_divisors(factorize(absolute_value(b)), 1);
    }

    std::vector<std::int64_t> roots;
    for (std::int64_t candidate : candidates) {
        if (cubic(a, b, candidate) == 0) {
            roots.push_back(candidate);
        }
    }
    std::sort(roots.begin(), roots.end());
    roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
    return roots;
}

std::vector<std::int64_t> three_torsion_x(std::int64_t a, std::int64_t b) {
    std::vector<std::int64_t> candidates;
    if (a == 0) {
        candidates = {0};
        const auto cubic_root = exact_integer_cube_root(-4 * static_cast<Int128>(b));
        if (cubic_root.has_value() && *cubic_root != 0) {
            candidates.push_back(*cubic_root);
        }
    } else {
        candidates = signed_divisors(factorize(absolute_value(a)), 2);
    }

    std::vector<std::int64_t> roots;
    for (std::int64_t candidate : candidates) {
        if (third_division_polynomial(a, b, candidate) == 0
            && is_positive_square(cubic(a, b, candidate))) {
            roots.push_back(candidate);
        }
    }
    std::sort(roots.begin(), roots.end());
    roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
    return roots;
}

std::string classify_case(bool has_three_torsion, std::size_t two_torsion_count) {
    if (has_three_torsion && two_torsion_count == 0) {
        return "A1";
    }
    if (has_three_torsion && two_torsion_count == 1) {
        return "A2";
    }
    if (has_three_torsion && two_torsion_count == 3) {
        return "A3";
    }
    if (!has_three_torsion && two_torsion_count == 0) {
        return "A4";
    }
    if (!has_three_torsion && two_torsion_count == 1) {
        return "A5";
    }
    if (!has_three_torsion && two_torsion_count == 3) {
        return "A6";
    }
    return "NA";
}

std::string join_roots(const std::vector<std::int64_t>& roots) {
    if (roots.empty()) {
        return "-";
    }
    std::ostringstream output;
    for (std::size_t index = 0; index < roots.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << roots[index];
    }
    return output.str();
}

bool is_nonsingular(std::int64_t a, std::int64_t b) {
    const Int128 coefficient_a = a;
    const Int128 coefficient_b = b;
    return 4 * coefficient_a * coefficient_a * coefficient_a
        + 27 * coefficient_b * coefficient_b != 0;
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream input(line);
        std::string sample_id;
        std::int64_t a = 0;
        std::int64_t b = 0;
        if (!(input >> sample_id >> a >> b)) {
            continue;
        }

        if (!is_nonsingular(a, b)) {
            std::cout << sample_id << "\t0\t0\t-\t-\n";
            continue;
        }

        const auto roots_two = two_torsion_roots(a, b);
        const auto roots_three = three_torsion_x(a, b);
        const bool has_three = !roots_three.empty();
        std::cout << sample_id << '\t'
                  << roots_two.size() << '\t'
                  << (has_three ? 1 : 0) << '\t'
                  << classify_case(has_three, roots_two.size()) << '\t'
                  << join_roots(roots_three) << '\n';
    }
}
